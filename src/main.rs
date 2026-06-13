use clap::Parser;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering as AtomicOrdering},
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

type Result<T> = std::result::Result<T, String>;
const GIT_VERSION: &str = env!("GIT_VERSION");

fn main() {
    println!(
        "qray {} ({GIT_VERSION}) build with {} ({})",
        env!("CARGO_PKG_VERSION"),
        env!("RUSTC_VERSION"),
        env!("BUILD_PROFILE")
    );
    if let Err(err) = run() {
        eprintln!("qray: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = Config::parse();
    config.validate()?;

    let started = Instant::now();
    if config.input.is_dir() {
        render_frame_directory(&config)?;
    } else {
        render_single_frame(&config)?;
    }

    if config.stats {
        eprintln!("total time: {:.2}s", started.elapsed().as_secs_f32());
    }

    Ok(())
}

#[derive(Debug, Clone, Parser)]
#[command(name = "qray", about = "Render qdqr POV frame files to PNG images")]
struct Config {
    #[arg(value_name = "FILE_OR_DIR")]
    input: PathBuf,

    #[arg(value_name = "OUTPUT")]
    output: Option<PathBuf>,

    #[arg(long, default_value_t = 320, help = "Output width in pixels")]
    width: usize,

    #[arg(long, default_value_t = 180, help = "Output height in pixels")]
    height: usize,

    #[arg(
        long,
        default_value_t = 8,
        help = "Nearest lights to shade per surface hit"
    )]
    max_lights: usize,

    #[arg(
        long = "no-shadows",
        action = clap::ArgAction::SetFalse,
        default_value_t = true,
        help = "Skip shadow rays"
    )]
    shadows: bool,

    #[arg(long, help = "Print parse and render statistics")]
    stats: bool,

    #[arg(long, help = "Sample PNG textures referenced by the scene files")]
    textures: bool,

    #[arg(
        long,
        help = "Smooth MDL object shading by averaging connected face normals"
    )]
    smooth_normals: bool,

    #[arg(long, help = "Enable adaptive antialiasing on high-contrast pixels")]
    aa: bool,

    #[arg(
        long,
        default_value_t = 4,
        help = "Maximum extra AA rays per triggered pixel, from 1 to 16"
    )]
    aa_samples: usize,

    #[arg(
        long,
        default_value_t = 0.12,
        help = "Linear color contrast threshold that triggers adaptive AA"
    )]
    aa_threshold: f32,

    #[arg(long, help = "First frame number for directory input")]
    start: Option<u32>,

    #[arg(long, help = "Last frame number for directory input")]
    end: Option<u32>,

    #[arg(long, help = "Maximum number of directory frames to render")]
    limit: Option<usize>,
}

impl Config {
    fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err("width and height must be non-zero".to_string());
        }
        if self.aa_samples == 0 || self.aa_samples > AA_OFFSETS.len() {
            return Err(format!(
                "aa-samples must be between 1 and {}",
                AA_OFFSETS.len()
            ));
        }
        if !self.aa_threshold.is_finite() || self.aa_threshold <= 0.0 {
            return Err("aa-threshold must be a positive finite number".to_string());
        }
        Ok(())
    }
}

fn render_single_frame(config: &Config) -> Result<()> {
    let output = config.output.clone().unwrap_or_else(|| {
        let mut out = config
            .input
            .file_stem()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("render"));
        out.set_extension("png");
        out
    });

    let mut library = PovLibrary::default();
    let frame_dir = config.input.parent().unwrap_or_else(|| Path::new(""));
    let mut caches = FrameCaches::new(frame_dir, config.textures);
    render_frame(&mut library, &mut caches, &config.input, &output, config)
}

fn render_frame_directory(config: &Config) -> Result<()> {
    let output_dir = config
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("renders"));
    fs::create_dir_all(&output_dir)
        .map_err(|err| format!("failed to create `{}`: {err}", output_dir.display()))?;

    let mut frames = collect_frames(&config.input)?;
    frames.retain(|frame| {
        let Some(number) = frame_number(frame) else {
            return true;
        };
        if let Some(start) = config.start {
            if number < start {
                return false;
            }
        }
        if let Some(end) = config.end {
            if number > end {
                return false;
            }
        }
        true
    });

    if let Some(limit) = config.limit {
        frames.truncate(limit);
    }

    if frames.is_empty() {
        return Err(format!(
            "no .pov frames found in `{}` for the requested range",
            config.input.display()
        ));
    }

    let mut library = PovLibrary::default();
    let mut caches = FrameCaches::new(&config.input, config.textures);
    for frame in frames {
        let mut output = output_dir.join(
            frame
                .file_stem()
                .ok_or_else(|| format!("invalid frame path `{}`", frame.display()))?,
        );
        output.set_extension("png");
        render_frame(&mut library, &mut caches, &frame, &output, config)?;
    }

    Ok(())
}

fn collect_frames(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut frames = Vec::new();
    for entry in
        fs::read_dir(dir).map_err(|err| format!("failed to read `{}`: {err}", dir.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read directory entry: {err}"))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("pov") {
            frames.push(path);
        }
    }
    frames.sort();
    Ok(frames)
}

fn frame_number(path: &Path) -> Option<u32> {
    let stem = path.file_stem()?.to_str()?;
    stem.strip_prefix("frame-")?.parse().ok()
}

#[derive(Default)]
struct FrameCaches {
    texture_cache: Option<TextureCache>,
    instances: HashMap<InstanceCacheKey, CachedInstance>,
    macro_generation: u64,
}

impl FrameCaches {
    fn new(texture_root: &Path, textures: bool) -> Self {
        Self {
            texture_cache: textures.then(|| TextureCache::new(texture_root)),
            instances: HashMap::new(),
            macro_generation: 0,
        }
    }
}

#[derive(Clone, Debug)]
struct CachedInstance {
    triangles: Vec<Triangle>,
    lights: Vec<Light>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct InstanceCacheKey {
    name: String,
    texture_arg: String,
    pos: [u32; 3],
    rot: [u32; 3],
    textures: bool,
    smooth_normals: bool,
}

impl InstanceCacheKey {
    fn new(call: &MacroCall, textures: bool, smooth_normals: bool) -> Self {
        Self {
            name: call.name.clone(),
            texture_arg: call.texture_arg.clone(),
            pos: vec3_bits(call.pos),
            rot: vec3_bits(call.rot),
            textures,
            smooth_normals: smooth_normals && call.is_mdl_object(),
        }
    }
}

fn vec3_bits(value: Vec3) -> [u32; 3] {
    [value.x.to_bits(), value.y.to_bits(), value.z.to_bits()]
}

fn render_frame(
    library: &mut PovLibrary,
    caches: &mut FrameCaches,
    input: &Path,
    output: &Path,
    config: &Config,
) -> Result<()> {
    let started = Instant::now();
    let build = build_scene(library, caches, input, config)?;
    if config.stats {
        eprintln!(
            "{}: {} calls, {} triangles, {} lights, {} textures, {} warnings, instance cache {} hits / {} misses, parse/build {:.2}s",
            input.display(),
            build.call_count,
            build.scene.triangles.len(),
            build.scene.lights.len(),
            build.texture_count,
            build.warnings.len(),
            build.instance_cache_hits,
            build.instance_cache_misses,
            started.elapsed().as_secs_f32()
        );
        for warning in build.warnings.iter().take(8) {
            eprintln!("  warning: {warning}");
        }
        if build.warnings.len() > 8 {
            eprintln!("  warning: {} more omitted", build.warnings.len() - 8);
        }
    }

    let render_started = Instant::now();
    let pixels = render(&build.scene, config);
    let render_elapsed = render_started.elapsed();
    let png_started = Instant::now();
    let ofilename = output.file_name().map(std::path::PathBuf::from);
    let metadata = PngMetadata::new(input, &ofilename.unwrap_or("".into()), config);
    write_png(output, config.width, config.height, &pixels, &metadata)?;
    let png_elapsed = png_started.elapsed();
    if config.stats {
        eprintln!(
            "{} -> {} render {:.2}s, png {:.2}s, total {:.2}s",
            input.display(),
            output.display(),
            render_elapsed.as_secs_f32(),
            png_elapsed.as_secs_f32(),
            render_started.elapsed().as_secs_f32()
        );
    }

    Ok(())
}

fn build_scene(
    library: &mut PovLibrary,
    caches: &mut FrameCaches,
    input: &Path,
    config: &Config,
) -> Result<SceneBuild> {
    let text = fs::read_to_string(input)
        .map_err(|err| format!("failed to read frame `{}`: {err}", input.display()))?;
    let frame_dir = input
        .parent()
        .ok_or_else(|| format!("frame `{}` has no parent directory", input.display()))?;

    let static_lights = library.load_includes(frame_dir, &text)?;
    if caches.macro_generation != library.macro_generation {
        caches.instances.clear();
        caches.macro_generation = library.macro_generation;
    }

    let mut warnings = std::mem::take(&mut library.warnings);
    let camera =
        parse_camera(&text).ok_or_else(|| format!("missing camera in `{}`", input.display()))?;
    let calls = parse_macro_calls(&text);

    let mut triangles = Vec::new();
    let mut lights = Vec::new();
    let mut instance_cache_hits = 0usize;
    let mut instance_cache_misses = 0usize;
    for call in &calls {
        if let Some(template) = library.macros.get(&call.name) {
            let key = InstanceCacheKey::new(call, config.textures, config.smooth_normals);
            if let Some(cached) = caches.instances.get(&key) {
                triangles.extend_from_slice(&cached.triangles);
                lights.extend_from_slice(&cached.lights);
                instance_cache_hits += 1;
                continue;
            }

            let triangle_start = triangles.len();
            let light_start = lights.len();
            template.instantiate(
                call,
                &mut triangles,
                &mut lights,
                &mut caches.texture_cache,
                config.smooth_normals,
            );
            caches.instances.insert(
                key,
                CachedInstance {
                    triangles: triangles[triangle_start..].to_vec(),
                    lights: lights[light_start..].to_vec(),
                },
            );
            instance_cache_misses += 1;
        } else {
            warnings.push(format!(
                "macro `{}` is referenced but not loaded",
                call.name
            ));
        }
    }
    for light in static_lights {
        lights.push(Light {
            position: light.position,
            color: light.color,
            intensity: light.intensity,
            fade_distance: light.fade_distance,
            fade_power: light.fade_power,
        });
    }

    if triangles.is_empty() {
        return Err(format!(
            "frame `{}` did not instantiate any triangles",
            input.display()
        ));
    }

    if lights.is_empty() {
        let pos = camera.location - camera.forward() * 32.0 + camera.up_dir() * 32.0;
        lights.push(Light {
            position: pos,
            color: Color::splat(1.0),
            intensity: 2.0,
            fade_distance: 128.0,
            fade_power: 1.0,
        });
        warnings.push("no scene lights found; inserted a camera light".to_string());
    }

    let textures = if let Some(cache) = caches.texture_cache.as_mut() {
        warnings.append(&mut cache.warnings);
        cache.textures.clone()
    } else {
        Vec::new()
    };
    let texture_count = textures.len();
    let scene = Scene::new(camera, triangles, lights, textures, config.stats);
    Ok(SceneBuild {
        scene,
        call_count: calls.len(),
        warnings,
        instance_cache_hits,
        instance_cache_misses,
        texture_count,
    })
}

struct SceneBuild {
    scene: Scene,
    call_count: usize,
    warnings: Vec<String>,
    instance_cache_hits: usize,
    instance_cache_misses: usize,
    texture_count: usize,
}

#[derive(Default)]
struct PovLibrary {
    macros: HashMap<String, MacroTemplate>,
    parsed_files: HashSet<PathBuf>,
    file_lights: HashMap<PathBuf, Vec<LightTemplate>>,
    warnings: Vec<String>,
    macro_generation: u64,
}

impl PovLibrary {
    fn load_includes(&mut self, frame_dir: &Path, frame_text: &str) -> Result<Vec<LightTemplate>> {
        let mut lights = Vec::new();
        let mut frame_files = HashSet::new();
        for include in parse_includes(frame_text) {
            let path = frame_dir.join(&include);
            if !path.exists() {
                if include != "rad_def.inc" {
                    self.warnings
                        .push(format!("include `{}` does not exist", path.display()));
                }
                continue;
            }
            let key = self.load_file(&path)?;
            if frame_files.insert(key.clone()) {
                if let Some(file_lights) = self.file_lights.get(&key) {
                    lights.extend(file_lights.iter().copied());
                }
            }
        }
        Ok(lights)
    }

    fn load_file(&mut self, path: &Path) -> Result<PathBuf> {
        let key = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if self.parsed_files.contains(&key) {
            return Ok(key);
        }
        self.parsed_files.insert(key.clone());

        let text = fs::read_to_string(path)
            .map_err(|err| format!("failed to read include `{}`: {err}", path.display()))?;
        for include in parse_includes(&text) {
            let nested = path
                .parent()
                .ok_or_else(|| format!("include `{}` has no parent directory", path.display()))?
                .join(&include);
            if nested.exists() {
                self.load_file(&nested)?;
            } else if include != "rad_def.inc" {
                self.warnings
                    .push(format!("include `{}` does not exist", nested.display()));
            }
        }

        for (name, body) in parse_macros(&text) {
            let template = parse_macro_template(&name, &body);
            self.macros.insert(name, template);
            self.macro_generation = self.macro_generation.wrapping_add(1);
        }
        self.file_lights
            .insert(key.clone(), parse_top_level_lights(&text));
        Ok(key)
    }
}

#[derive(Clone, Debug)]
struct MacroCall {
    name: String,
    pos: Vec3,
    rot: Vec3,
    texture_arg: String,
}

impl MacroCall {
    fn is_mdl_object(&self) -> bool {
        self.texture_arg.contains(".mdl") || self.name.contains("_mdl_")
    }
}

#[derive(Clone, Debug, Default)]
struct MacroTemplate {
    meshes: Vec<MeshTemplate>,
    lights: Vec<LightTemplate>,
}

impl MacroTemplate {
    fn instantiate(
        &self,
        call: &MacroCall,
        triangles: &mut Vec<Triangle>,
        lights: &mut Vec<Light>,
        texture_cache: &mut Option<TextureCache>,
        smooth_normals: bool,
    ) {
        for mesh in &self.meshes {
            mesh.instantiate(call, triangles, texture_cache, smooth_normals);
        }
        for light in &self.lights {
            lights.push(Light {
                position: transform_point(light.position, call.rot, call.pos),
                color: light.color,
                intensity: light.intensity,
                fade_distance: light.fade_distance,
                fade_power: light.fade_power,
            });
        }
    }
}

#[derive(Clone, Debug)]
struct MeshTemplate {
    vertices: Vec<Vec3>,
    uvs: Vec<Vec2>,
    faces: Vec<FaceTemplate>,
    uv_indices: Vec<[usize; 3]>,
    materials: Vec<MaterialTemplate>,
}

impl MeshTemplate {
    fn instantiate(
        &self,
        call: &MacroCall,
        triangles: &mut Vec<Triangle>,
        texture_cache: &mut Option<TextureCache>,
        smooth_normals: bool,
    ) {
        if self.vertices.is_empty() || self.faces.is_empty() {
            return;
        }

        let transformed: Vec<Vec3> = self
            .vertices
            .iter()
            .map(|&vertex| transform_point(vertex, call.rot, call.pos))
            .collect();
        let vertex_normals = (smooth_normals && call.is_mdl_object())
            .then(|| compute_vertex_normals(&transformed, &self.faces));

        let texture_indices: Vec<Option<usize>> = self
            .materials
            .iter()
            .map(|material| {
                texture_cache.as_mut().and_then(|cache| {
                    material
                        .texture_path(&call.texture_arg)
                        .and_then(|path| cache.texture_index(path))
                })
            })
            .collect();

        for (face_index, face) in self.faces.iter().enumerate() {
            let Some(v0) = transformed.get(face.indices[0]).copied() else {
                continue;
            };
            let Some(v1) = transformed.get(face.indices[1]).copied() else {
                continue;
            };
            let Some(v2) = transformed.get(face.indices[2]).copied() else {
                continue;
            };
            let color = self
                .materials
                .get(face.material_index)
                .map(|material| material.resolve(&call.texture_arg))
                .unwrap_or_else(|| material_color_from_key(&call.name));
            let uv = self
                .uv_indices
                .get(face_index)
                .and_then(|indices| uv_triangle(&self.uvs, *indices));
            let texture_index = texture_indices
                .get(face.material_index)
                .copied()
                .flatten()
                .filter(|_| uv.is_some());
            let normals = vertex_normals.as_ref().map(|vertex_normals| {
                [
                    vertex_normals
                        .get(face.indices[0])
                        .copied()
                        .unwrap_or_default(),
                    vertex_normals
                        .get(face.indices[1])
                        .copied()
                        .unwrap_or_default(),
                    vertex_normals
                        .get(face.indices[2])
                        .copied()
                        .unwrap_or_default(),
                ]
            });
            if let Some(triangle) = Triangle::new(v0, v1, v2, color, texture_index, uv, normals) {
                triangles.push(triangle);
            }
        }
    }
}

fn compute_vertex_normals(vertices: &[Vec3], faces: &[FaceTemplate]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::default(); vertices.len()];

    for face in faces {
        let Some(v0) = vertices.get(face.indices[0]).copied() else {
            continue;
        };
        let Some(v1) = vertices.get(face.indices[1]).copied() else {
            continue;
        };
        let Some(v2) = vertices.get(face.indices[2]).copied() else {
            continue;
        };

        let face_normal = (v1 - v0).cross(v2 - v0).normalize_or(Vec3::default());
        if face_normal.length() < 1.0e-8 {
            continue;
        }

        for &index in &face.indices {
            if let Some(normal) = normals.get_mut(index) {
                *normal += face_normal;
            }
        }
    }

    normals
        .into_iter()
        .map(|normal| normal.normalize_or(Vec3::default()))
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct FaceTemplate {
    indices: [usize; 3],
    material_index: usize,
}

#[derive(Clone, Debug)]
enum MaterialTemplate {
    Color(Color),
    Key(String),
    Texture { key: String, source: TextureSource },
}

#[derive(Clone, Debug)]
enum TextureSource {
    TexturePrefixSuffix(String),
    Skin,
    File(String),
}

impl MaterialTemplate {
    fn resolve(&self, texture_arg: &str) -> Color {
        match self {
            Self::Color(color) => *color,
            Self::Key(key) => material_color_from_key(key),
            Self::Texture { key, source } => match source {
                TextureSource::Skin => material_color_from_key(texture_arg),
                _ => material_color_from_key(key),
            },
        }
    }

    fn texture_path(&self, texture_arg: &str) -> Option<PathBuf> {
        let Self::Texture { source, .. } = self else {
            return None;
        };
        Some(match source {
            TextureSource::TexturePrefixSuffix(suffix) => PathBuf::from(texture_arg)
                .join(suffix.trim_start_matches('/').trim_start_matches('\\')),
            TextureSource::Skin => PathBuf::from(texture_arg),
            TextureSource::File(path) => PathBuf::from(path),
        })
    }
}

#[derive(Debug)]
struct TextureCache {
    root: PathBuf,
    paths: HashMap<PathBuf, usize>,
    failed: HashSet<PathBuf>,
    textures: Vec<Arc<Texture>>,
    warnings: Vec<String>,
}

impl TextureCache {
    fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            paths: HashMap::new(),
            failed: HashSet::new(),
            textures: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn texture_index(&mut self, path: PathBuf) -> Option<usize> {
        let path = if path.is_absolute() {
            path
        } else {
            self.root.join(path)
        };
        let key = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if let Some(&index) = self.paths.get(&key) {
            return Some(index);
        }
        if self.failed.contains(&key) {
            return None;
        }

        match Texture::load(&path) {
            Ok(texture) => {
                let index = self.textures.len();
                self.textures.push(Arc::new(texture));
                self.paths.insert(key, index);
                Some(index)
            }
            Err(err) => {
                self.failed.insert(key);
                self.warnings.push(format!(
                    "texture `{}` could not be loaded: {err}; using flat material color",
                    path.display()
                ));
                None
            }
        }
    }
}

#[derive(Debug)]
struct Texture {
    width: usize,
    height: usize,
    pixels: Vec<Color>,
}

impl Texture {
    fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|err| err.to_string())?;
        let mut decoder = png::Decoder::new(BufReader::new(file));
        decoder.set_transformations(png::Transformations::normalize_to_color8());
        let mut reader = decoder.read_info().map_err(|err| err.to_string())?;
        let mut buffer = vec![0; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut buffer)
            .map_err(|err| err.to_string())?;
        let bytes = &buffer[..info.buffer_size()];
        let pixels = decode_texture_pixels(info.color_type, bytes)?;

        Ok(Self {
            width: info.width as usize,
            height: info.height as usize,
            pixels,
        })
    }

    fn sample(&self, uv: Vec2) -> Color {
        if self.width == 0 || self.height == 0 || self.pixels.is_empty() {
            return Color::splat(1.0);
        }

        let x = uv.x.rem_euclid(1.0) * self.width as f32 - 0.5;
        let y = uv.y.rem_euclid(1.0) * self.height as f32 - 0.5;
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let tx = x - x.floor();
        let ty = y - y.floor();

        let c00 = self.pixel_wrapped(x0, y0);
        let c10 = self.pixel_wrapped(x1, y0);
        let c01 = self.pixel_wrapped(x0, y1);
        let c11 = self.pixel_wrapped(x1, y1);
        let top = lerp_color(c00, c10, tx);
        let bottom = lerp_color(c01, c11, tx);
        lerp_color(top, bottom, ty)
    }

    fn pixel_wrapped(&self, x: isize, y: isize) -> Color {
        let x = x.rem_euclid(self.width as isize) as usize;
        let y = y.rem_euclid(self.height as isize) as usize;
        self.pixels[y * self.width + x]
    }
}

fn decode_texture_pixels(color_type: png::ColorType, bytes: &[u8]) -> Result<Vec<Color>> {
    match color_type {
        png::ColorType::Rgb => Ok(bytes
            .chunks_exact(3)
            .map(|pixel| srgb8(pixel[0], pixel[1], pixel[2]))
            .collect()),
        png::ColorType::Rgba => Ok(bytes
            .chunks_exact(4)
            .map(|pixel| srgb8(pixel[0], pixel[1], pixel[2]))
            .collect()),
        png::ColorType::Grayscale => {
            Ok(bytes.iter().map(|&gray| srgb8(gray, gray, gray)).collect())
        }
        png::ColorType::GrayscaleAlpha => Ok(bytes
            .chunks_exact(2)
            .map(|pixel| srgb8(pixel[0], pixel[0], pixel[0]))
            .collect()),
        png::ColorType::Indexed => Err("indexed PNG was not expanded by the decoder".to_string()),
    }
}

fn srgb8(r: u8, g: u8, b: u8) -> Color {
    Color::new(
        srgb_byte_to_linear(r),
        srgb_byte_to_linear(g),
        srgb_byte_to_linear(b),
    )
}

fn srgb_byte_to_linear(value: u8) -> f32 {
    let value = value as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    a * (1.0 - t) + b * t
}

#[derive(Clone, Copy, Debug)]
struct LightTemplate {
    position: Vec3,
    color: Color,
    intensity: f32,
    fade_distance: f32,
    fade_power: f32,
}

#[derive(Clone, Copy, Debug)]
struct Light {
    position: Vec3,
    color: Color,
    intensity: f32,
    fade_distance: f32,
    fade_power: f32,
}

#[derive(Clone, Copy, Debug)]
struct Camera {
    angle: f32,
    location: Vec3,
    look_at: Vec3,
    up: Vec3,
    right: Vec3,
    sky: Vec3,
}

impl Camera {
    fn forward(&self) -> Vec3 {
        (self.look_at - self.location).normalize_or(Vec3::new(1.0, 0.0, 0.0))
    }

    fn up_dir(&self) -> Vec3 {
        self.basis().2
    }

    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = self.forward();
        let up_hint = self
            .up
            .normalize_or(self.sky.normalize_or(Vec3::new(0.0, 0.0, 1.0)));
        let mut right = forward.cross(up_hint).normalize_or(Vec3::default());
        if right.length() < 1.0e-8 {
            let fallback = if forward.z.abs() < 0.9 {
                Vec3::new(0.0, 0.0, 1.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            right = forward
                .cross(fallback)
                .normalize_or(Vec3::new(1.0, 0.0, 0.0));
        }
        let up = right.cross(forward).normalize_or(up_hint);
        (forward, right, up)
    }

    fn ray_for_pixel(&self, x: usize, y: usize, width: usize, height: usize) -> Ray {
        self.ray_for_sample(x as f32 + 0.5, y as f32 + 0.5, width, height)
    }

    fn ray_for_sample(&self, sample_x: f32, sample_y: f32, width: usize, height: usize) -> Ray {
        let (forward, right, up) = self.basis();

        let aspect = if self.right.length() > 1.0e-8 && self.up.length() > 1.0e-8 {
            self.right.length() / self.up.length()
        } else {
            width as f32 / height as f32
        };
        let angle = self.angle.clamp(1.0, 175.0) * PI / 180.0;
        let viewport_width = 2.0 * (angle * 0.5).tan();
        let viewport_height = viewport_width / aspect;
        let px = (sample_x / width as f32 - 0.5) * viewport_width;
        let py = (0.5 - sample_y / height as f32) * viewport_height;

        Ray {
            origin: self.location,
            dir: (forward + right * px + up * py).normalize_or(forward),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn splat(value: f32) -> Self {
        Self::new(value, value, value)
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    fn normalize_or(self, fallback: Self) -> Self {
        let length = self.length();
        if length > 1.0e-8 {
            self / length
        } else {
            fallback
        }
    }

    fn min(self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    fn max(self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    fn component(self, axis: usize) -> f32 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Color {
    r: f32,
    g: f32,
    b: f32,
}

impl Color {
    const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    fn splat(value: f32) -> Self {
        Self::new(value, value, value)
    }

    fn clamp01(self) -> Self {
        Self::new(
            self.r.clamp(0.0, 1.0),
            self.g.clamp(0.0, 1.0),
            self.b.clamp(0.0, 1.0),
        )
    }

    fn tone_map(self) -> Self {
        Self::new(
            self.r / (1.0 + self.r),
            self.g / (1.0 + self.g),
            self.b / (1.0 + self.b),
        )
    }
}

impl std::ops::Add for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}

impl std::ops::AddAssign for Color {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Mul<f32> for Color {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.r * rhs, self.g * rhs, self.b * rhs)
    }
}

impl std::ops::Mul for Color {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.r * rhs.r, self.g * rhs.g, self.b * rhs.b)
    }
}

#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vec3,
    dir: Vec3,
}

#[derive(Debug)]
struct Scene {
    camera: Camera,
    triangles: Vec<Triangle>,
    lights: Vec<Light>,
    textures: Vec<Arc<Texture>>,
    bvh: BvhNode,
}

impl Scene {
    fn new(
        camera: Camera,
        triangles: Vec<Triangle>,
        lights: Vec<Light>,
        textures: Vec<Arc<Texture>>,
        stats: bool,
    ) -> Self {
        let start = Instant::now();
        let mut indices: Vec<usize> = (0..triangles.len()).collect();
        let bvh = BvhNode::build(&mut indices, &triangles);
        if stats {
            eprintln!("BVH build: {:.2}s", start.elapsed().as_secs_f32());
        }
        Self {
            camera,
            triangles,
            lights,
            textures,
            bvh,
        }
    }

    fn hit(&self, ray: Ray, max_t: f32) -> Option<HitRecord> {
        let mut best = HitRecord {
            t: max_t,
            triangle_index: usize::MAX,
            bary_u: 0.0,
            bary_v: 0.0,
        };
        self.bvh.hit(ray, &self.triangles, &mut best);
        if best.triangle_index == usize::MAX {
            None
        } else {
            Some(best)
        }
    }

    fn occluded(&self, ray: Ray, max_t: f32) -> bool {
        self.bvh.occluded(ray, &self.triangles, max_t)
    }
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    v0: Vec3,
    e1: Vec3,
    e2: Vec3,
    normal: Vec3,
    vertex_normals: [Vec3; 3],
    color: Color,
    texture_index: Option<usize>,
    uv: Option<[Vec2; 3]>,
    bbox: Aabb,
    centroid: Vec3,
}

impl Triangle {
    fn new(
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        color: Color,
        texture_index: Option<usize>,
        uv: Option<[Vec2; 3]>,
        vertex_normals: Option<[Vec3; 3]>,
    ) -> Option<Self> {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let normal = e1.cross(e2).normalize_or(Vec3::default());
        if normal.length() < 1.0e-8 {
            return None;
        }
        let vertex_normals = vertex_normals
            .map(|normals| {
                [
                    normals[0].normalize_or(normal),
                    normals[1].normalize_or(normal),
                    normals[2].normalize_or(normal),
                ]
            })
            .unwrap_or([normal; 3]);
        let mut bbox = Aabb::empty();
        bbox.grow(v0);
        bbox.grow(v1);
        bbox.grow(v2);
        bbox.pad(1.0e-4);

        Some(Self {
            v0,
            e1,
            e2,
            normal,
            vertex_normals,
            color,
            texture_index,
            uv,
            bbox,
            centroid: (v0 + v1 + v2) / 3.0,
        })
    }

    fn intersect(&self, ray: Ray, max_t: f32) -> Option<(f32, f32, f32)> {
        let p = ray.dir.cross(self.e2);
        let det = self.e1.dot(p);
        if det.abs() < 1.0e-7 {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = ray.origin - self.v0;
        let u = tvec.dot(p) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = tvec.cross(self.e1);
        let v = ray.dir.dot(q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = self.e2.dot(q) * inv_det;
        if t > 1.0e-4 && t < max_t {
            Some((t, u, v))
        } else {
            None
        }
    }

    fn color_at(&self, bary_u: f32, bary_v: f32, textures: &[Arc<Texture>]) -> Color {
        let Some(texture_index) = self.texture_index else {
            return self.color;
        };
        let Some(uv) = self.uv else {
            return self.color;
        };
        let Some(texture) = textures.get(texture_index) else {
            return self.color;
        };

        let bary_w = 1.0 - bary_u - bary_v;
        let uv = uv[0] * bary_w + uv[1] * bary_u + uv[2] * bary_v;
        texture.sample(uv)
    }

    fn normal_at(&self, bary_u: f32, bary_v: f32) -> Vec3 {
        let bary_w = 1.0 - bary_u - bary_v;
        (self.vertex_normals[0] * bary_w
            + self.vertex_normals[1] * bary_u
            + self.vertex_normals[2] * bary_v)
            .normalize_or(self.normal)
    }
}

#[derive(Clone, Copy, Debug)]
struct HitRecord {
    t: f32,
    triangle_index: usize,
    bary_u: f32,
    bary_v: f32,
}

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    fn grow(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn pad(&mut self, amount: f32) {
        let pad = Vec3::splat(amount);
        self.min = self.min - pad;
        self.max += pad;
    }

    fn longest_axis(self) -> usize {
        let extent = self.max - self.min;
        if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        }
    }

    #[cfg(test)]
    fn hit(self, ray: Ray, max_t: f32) -> bool {
        self.hit_distance(ray, max_t).is_some()
    }

    fn hit_distance(self, ray: Ray, max_t: f32) -> Option<f32> {
        let mut t_min: f32 = 0.0;
        let mut t_max: f32 = max_t;

        for axis in 0..3 {
            let origin = ray.origin.component(axis);
            let dir = ray.dir.component(axis);
            let min = self.min.component(axis);
            let max = self.max.component(axis);
            if dir.abs() < 1.0e-8 {
                if origin < min || origin > max {
                    return None;
                }
                continue;
            }

            let inv = 1.0 / dir;
            let mut t0 = (min - origin) * inv;
            let mut t1 = (max - origin) * inv;
            if inv < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
            if t_max < t_min {
                return None;
            }
        }

        Some(t_min)
    }
}

#[derive(Debug)]
enum BvhNode {
    Leaf {
        bbox: Aabb,
        triangles: Vec<usize>,
    },
    Branch {
        bbox: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

impl BvhNode {
    fn build(indices: &mut [usize], triangles: &[Triangle]) -> Self {
        let bbox = indices.iter().fold(Aabb::empty(), |bbox, &index| {
            bbox.union(triangles[index].bbox)
        });

        if indices.len() <= 8 {
            return Self::Leaf {
                bbox,
                triangles: indices.to_vec(),
            };
        }

        let mut centroid_bounds = Aabb::empty();
        for &index in indices.iter() {
            centroid_bounds.grow(triangles[index].centroid);
        }
        let axis = centroid_bounds.longest_axis();
        indices.sort_by(|&a, &b| {
            triangles[a]
                .centroid
                .component(axis)
                .partial_cmp(&triangles[b].centroid.component(axis))
                .unwrap_or(Ordering::Equal)
        });
        let split = indices.len() / 2;
        let (left, right) = indices.split_at_mut(split);

        Self::Branch {
            bbox,
            left: Box::new(Self::build(left, triangles)),
            right: Box::new(Self::build(right, triangles)),
        }
    }

    fn hit(&self, ray: Ray, triangles: &[Triangle], best: &mut HitRecord) {
        if self.bbox().hit_distance(ray, best.t).is_none() {
            return;
        }
        self.hit_after_bbox(ray, triangles, best);
    }

    fn hit_after_bbox(&self, ray: Ray, triangles: &[Triangle], best: &mut HitRecord) {
        match self {
            Self::Leaf {
                triangles: leaf, ..
            } => {
                for &index in leaf {
                    if let Some((t, u, v)) = triangles[index].intersect(ray, best.t) {
                        best.t = t;
                        best.triangle_index = index;
                        best.bary_u = u;
                        best.bary_v = v;
                    }
                }
            }
            Self::Branch { left, right, .. } => {
                let left_distance = left.bbox().hit_distance(ray, best.t);
                let right_distance = right.bbox().hit_distance(ray, best.t);
                match (left_distance, right_distance) {
                    (Some(left_t), Some(right_t)) if right_t < left_t => {
                        right.hit_after_bbox(ray, triangles, best);
                        if left_t <= best.t {
                            left.hit_after_bbox(ray, triangles, best);
                        }
                    }
                    (Some(_), Some(right_t)) => {
                        left.hit_after_bbox(ray, triangles, best);
                        if right_t <= best.t {
                            right.hit_after_bbox(ray, triangles, best);
                        }
                    }
                    (Some(_), None) => left.hit_after_bbox(ray, triangles, best),
                    (None, Some(_)) => right.hit_after_bbox(ray, triangles, best),
                    (None, None) => {}
                }
            }
        }
    }

    fn occluded(&self, ray: Ray, triangles: &[Triangle], max_t: f32) -> bool {
        if self.bbox().hit_distance(ray, max_t).is_none() {
            return false;
        }
        self.occluded_after_bbox(ray, triangles, max_t)
    }

    fn occluded_after_bbox(&self, ray: Ray, triangles: &[Triangle], max_t: f32) -> bool {
        match self {
            Self::Leaf {
                triangles: leaf, ..
            } => leaf
                .iter()
                .any(|&index| triangles[index].intersect(ray, max_t).is_some()),
            Self::Branch { left, right, .. } => {
                let left_distance = left.bbox().hit_distance(ray, max_t);
                let right_distance = right.bbox().hit_distance(ray, max_t);
                match (left_distance, right_distance) {
                    (Some(left_t), Some(right_t)) if right_t < left_t => {
                        right.occluded_after_bbox(ray, triangles, max_t)
                            || left.occluded_after_bbox(ray, triangles, max_t)
                    }
                    (Some(_), Some(_)) => {
                        left.occluded_after_bbox(ray, triangles, max_t)
                            || right.occluded_after_bbox(ray, triangles, max_t)
                    }
                    (Some(_), None) => left.occluded_after_bbox(ray, triangles, max_t),
                    (None, Some(_)) => right.occluded_after_bbox(ray, triangles, max_t),
                    (None, None) => false,
                }
            }
        }
    }

    fn bbox(&self) -> Aabb {
        match self {
            Self::Leaf { bbox, .. } | Self::Branch { bbox, .. } => *bbox,
        }
    }
}

fn render(scene: &Scene, config: &Config) -> Vec<Color> {
    let mut pixels = vec![Color::default(); config.width * config.height];
    pixels
        .par_chunks_mut(config.width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, pixel) in row.iter_mut().enumerate() {
                let ray = scene
                    .camera
                    .ray_for_pixel(x, y, config.width, config.height);
                *pixel = trace(scene, ray, config);
            }
        });

    if config.aa {
        adaptive_antialias(scene, config, &pixels)
    } else {
        pixels
    }
}

const AA_OFFSETS: [(f32, f32); 16] = [
    (0.25, 0.25),
    (0.75, 0.25),
    (0.25, 0.75),
    (0.75, 0.75),
    (0.50, 0.125),
    (0.875, 0.50),
    (0.50, 0.875),
    (0.125, 0.50),
    (0.125, 0.125),
    (0.875, 0.125),
    (0.125, 0.875),
    (0.875, 0.875),
    (0.375, 0.375),
    (0.625, 0.375),
    (0.375, 0.625),
    (0.625, 0.625),
];

fn adaptive_antialias(scene: &Scene, config: &Config, primary: &[Color]) -> Vec<Color> {
    let mut pixels = primary.to_vec();
    let aa_pixels = AtomicUsize::new(0);
    let aa_rays = AtomicUsize::new(0);

    pixels
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, pixel)| {
            let x = index % config.width;
            let y = index / config.width;
            if !should_antialias(
                primary,
                x,
                y,
                config.width,
                config.height,
                config.aa_threshold,
            ) {
                return;
            }

            *pixel = supersample_pixel(scene, config, x, y, primary[index]);
            aa_pixels.fetch_add(1, AtomicOrdering::Relaxed);
            aa_rays.fetch_add(config.aa_samples, AtomicOrdering::Relaxed);
        });

    if config.stats {
        let aa_pixels = aa_pixels.load(AtomicOrdering::Relaxed);
        let aa_rays = aa_rays.load(AtomicOrdering::Relaxed);
        eprintln!(
            "adaptive AA: {} / {} pixels, {} extra rays",
            aa_pixels,
            config.width * config.height,
            aa_rays
        );
    }

    pixels
}

fn should_antialias(
    pixels: &[Color],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    threshold: f32,
) -> bool {
    let center = pixels[y * width + x];
    let min_x = x.saturating_sub(1);
    let max_x = (x + 1).min(width - 1);
    let min_y = y.saturating_sub(1);
    let max_y = (y + 1).min(height - 1);

    for yy in min_y..=max_y {
        for xx in min_x..=max_x {
            if xx == x && yy == y {
                continue;
            }
            if color_contrast(center, pixels[yy * width + xx]) >= threshold {
                return true;
            }
        }
    }
    false
}

fn supersample_pixel(scene: &Scene, config: &Config, x: usize, y: usize, center: Color) -> Color {
    let mut color = center;
    for &(offset_x, offset_y) in AA_OFFSETS.iter().take(config.aa_samples) {
        let ray = scene.camera.ray_for_sample(
            x as f32 + offset_x,
            y as f32 + offset_y,
            config.width,
            config.height,
        );
        color += trace(scene, ray, config);
    }
    color * (1.0 / (config.aa_samples as f32 + 1.0))
}

fn color_contrast(a: Color, b: Color) -> f32 {
    (a.r - b.r)
        .abs()
        .max((a.g - b.g).abs())
        .max((a.b - b.b).abs())
}

fn trace(scene: &Scene, ray: Ray, config: &Config) -> Color {
    let Some(hit) = scene.hit(ray, f32::INFINITY) else {
        return background_color(ray.dir);
    };

    let triangle = &scene.triangles[hit.triangle_index];
    let base_color = triangle.color_at(hit.bary_u, hit.bary_v, &scene.textures);
    let point = ray.origin + ray.dir * hit.t;
    let mut normal = triangle.normal_at(hit.bary_u, hit.bary_v);
    if normal.dot(ray.dir) > 0.0 {
        normal = -normal;
    }
    let mut bias_normal = triangle.normal;
    if bias_normal.dot(ray.dir) > 0.0 {
        bias_normal = -bias_normal;
    }

    let mut color = base_color * 0.14;
    color += base_color * (0.08 * normal.z.max(0.0));
    color += base_color * (0.05 * (-ray.dir).dot(normal).max(0.0));

    for_nearest_lights(point, &scene.lights, config.max_lights, |index| {
        let light = scene.lights[index];
        let to_light = light.position - point;
        let distance = to_light.length();
        if distance < 1.0e-4 {
            return;
        }
        let light_dir = to_light / distance;
        let ndotl = normal.dot(light_dir).max(0.0);
        if ndotl <= 0.0 {
            return;
        }

        if config.shadows {
            let shadow_ray = Ray {
                origin: point + bias_normal * 0.03,
                dir: light_dir,
            };
            if scene.occluded(shadow_ray, distance - 0.06) {
                return;
            }
        }

        let attenuation = light_attenuation(distance, light.fade_distance, light.fade_power);
        let strength = ndotl * light.intensity * attenuation;
        color += base_color * light.color * strength;
    });

    color.tone_map().clamp01()
}

const STACK_LIGHT_LIMIT: usize = 32;

fn for_nearest_lights<F>(point: Vec3, lights: &[Light], limit: usize, mut each: F)
where
    F: FnMut(usize),
{
    if limit == 0 || lights.is_empty() {
        return;
    }
    if lights.len() <= limit {
        for index in 0..lights.len() {
            each(index);
        }
        return;
    }

    if limit <= STACK_LIGHT_LIMIT {
        let mut distances = [0.0; STACK_LIGHT_LIMIT];
        let mut indices = [0usize; STACK_LIGHT_LIMIT];
        let mut len = 0usize;

        for (index, light) in lights.iter().enumerate() {
            let delta = light.position - point;
            let distance2 = delta.dot(delta);
            if len < limit {
                distances[len] = distance2;
                indices[len] = index;
                len += 1;
                continue;
            }
            let mut farthest_slot = 0;
            for slot in 1..len {
                if distances[slot] > distances[farthest_slot] {
                    farthest_slot = slot;
                }
            }
            if distance2 < distances[farthest_slot] {
                distances[farthest_slot] = distance2;
                indices[farthest_slot] = index;
            }
        }

        for &index in indices.iter().take(len) {
            each(index);
        }
        return;
    }

    let mut nearest: Vec<(f32, usize)> = Vec::with_capacity(limit);
    for (index, light) in lights.iter().enumerate() {
        let delta = light.position - point;
        let distance2 = delta.dot(delta);
        if nearest.len() < limit {
            nearest.push((distance2, index));
            continue;
        }
        let mut farthest_slot = 0;
        for slot in 1..nearest.len() {
            if nearest[slot].0 > nearest[farthest_slot].0 {
                farthest_slot = slot;
            }
        }
        if distance2 < nearest[farthest_slot].0 {
            nearest[farthest_slot] = (distance2, index);
        }
    }

    for (_, index) in nearest {
        each(index);
    }
}

fn light_attenuation(distance: f32, fade_distance: f32, fade_power: f32) -> f32 {
    let fade_distance = fade_distance.max(1.0) * 8.0;
    let fade_power = fade_power.max(0.5);
    1.0 / (1.0 + (distance / fade_distance).powf(fade_power))
}

fn background_color(dir: Vec3) -> Color {
    let t = (dir.z * 0.5 + 0.5).clamp(0.0, 1.0);
    Color::new(0.02, 0.025, 0.035) * (1.0 - t) + Color::new(0.08, 0.11, 0.16) * t
}

#[derive(Debug)]
struct PngMetadata {
    input: String,
    output: String,
    render_parameters: String,
    timestamp_unix: String,
}

impl PngMetadata {
    fn new(input: &Path, output: &Path, config: &Config) -> Self {
        Self {
            input: input.display().to_string(),
            output: output.display().to_string(),
            render_parameters: config.render_parameters(),
            timestamp_unix: render_timestamp_unix(),
        }
    }
}

impl Config {
    fn render_parameters(&self) -> String {
        format!(
            "width={}, height={}, max_lights={}, shadows={}, textures={}, smooth_normals={}, aa={}, aa_samples={}, aa_threshold={}, start={}, end={}, limit={}",
            self.width,
            self.height,
            self.max_lights,
            self.shadows,
            self.textures,
            self.smooth_normals,
            self.aa,
            self.aa_samples,
            self.aa_threshold,
            option_u32(self.start),
            option_u32(self.end),
            option_usize(self.limit)
        )
    }
}

fn option_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn option_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn render_timestamp_unix() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn write_png(
    path: &Path,
    width: usize,
    height: usize,
    pixels: &[Color],
    metadata: &PngMetadata,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create `{}`: {err}", parent.display()))?;
        }
    }
    let file = File::create(path)
        .map_err(|err| format!("failed to create output `{}`: {err}", path.display()))?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Best);
    encoder.set_adaptive_filter(png::AdaptiveFilterType::Adaptive);
    add_png_text(
        &mut encoder,
        "Software",
        &format!("qray {GIT_VERSION}"),
        path,
    )?;
    add_png_text(
        &mut encoder,
        "qray.cargo_pkg_version",
        env!("CARGO_PKG_VERSION"),
        path,
    )?;
    add_png_text(&mut encoder, "qray.git_version", GIT_VERSION, path)?;
    add_png_text(
        &mut encoder,
        "qray.render_timestamp_unix",
        &metadata.timestamp_unix,
        path,
    )?;
    add_png_text(&mut encoder, "qray.input", &metadata.input, path)?;
    add_png_text(&mut encoder, "qray.output", &metadata.output, path)?;
    add_png_text(
        &mut encoder,
        "qray.render_parameters",
        &metadata.render_parameters,
        path,
    )?;

    let mut bytes = Vec::with_capacity(width * height * 3);
    for &color in pixels {
        bytes.push(to_png_byte(color.r));
        bytes.push(to_png_byte(color.g));
        bytes.push(to_png_byte(color.b));
    }
    let mut writer = encoder
        .write_header()
        .map_err(|err| format!("failed to write PNG header `{}`: {err}", path.display()))?;
    writer
        .write_image_data(&bytes)
        .map_err(|err| format!("failed to write PNG data `{}`: {err}", path.display()))?;
    Ok(())
}

fn add_png_text<W: std::io::Write>(
    encoder: &mut png::Encoder<W>,
    key: &str,
    value: &str,
    path: &Path,
) -> Result<()> {
    encoder
        .add_text_chunk(key.to_string(), value.to_string())
        .map_err(|err| format!("failed to add PNG metadata `{}`: {err}", path.display()))
}

fn to_png_byte(value: f32) -> u8 {
    let gamma = value.clamp(0.0, 1.0).powf(1.0 / 2.2);
    (gamma * 255.0 + 0.5) as u8
}

fn parse_macro_template(_name: &str, body: &str) -> MacroTemplate {
    let meshes = find_mesh2_blocks(body)
        .into_iter()
        .filter_map(|mesh| parse_mesh_template(mesh))
        .collect();
    let lights = parse_lights(body);
    MacroTemplate { meshes, lights }
}

fn parse_top_level_lights(text: &str) -> Vec<LightTemplate> {
    let mut lights = Vec::new();
    let mut pos = 0;
    while let Some(relative) = text[pos..].find("#macro") {
        let start = pos + relative;
        lights.extend(parse_lights(&text[pos..start]));

        let header_end = text[start..]
            .find('\n')
            .map(|end| start + end)
            .unwrap_or(text.len());
        let Some(end_relative) = text[header_end..].find("#end") else {
            return lights;
        };
        pos = header_end + end_relative + "#end".len();
    }
    if pos < text.len() {
        lights.extend(parse_lights(&text[pos..]));
    }
    lights
}

fn parse_mesh_template(mesh: &str) -> Option<MeshTemplate> {
    let vertices_block = extract_named_braced(mesh, "vertex_vectors")?;
    let vertices = parse_vec3_list(&mesh[vertices_block.0 + 1..vertices_block.1]);

    let uvs = extract_named_braced(mesh, "uv_vectors")
        .map(|(open, close)| parse_vec2_list(&mesh[open + 1..close]))
        .unwrap_or_default();

    let face_block = extract_named_braced(mesh, "face_indices")?;
    let faces = parse_face_indices(&mesh[face_block.0 + 1..face_block.1]);

    let uv_indices = extract_named_braced(mesh, "uv_indices")
        .map(|(open, close)| parse_index_triples(&mesh[open + 1..close]))
        .unwrap_or_default();

    let materials = extract_named_braced(mesh, "texture_list")
        .map(|(open, close)| parse_texture_list(&mesh[open + 1..close]))
        .unwrap_or_else(|| vec![MaterialTemplate::Key("default".to_string())]);

    Some(MeshTemplate {
        vertices,
        uvs,
        faces,
        uv_indices,
        materials,
    })
}

fn parse_includes(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let rest = line.strip_prefix("#include")?.trim();
            let first_quote = rest.find('"')?;
            let after = &rest[first_quote + 1..];
            let second_quote = after.find('"')?;
            Some(after[..second_quote].to_string())
        })
        .collect()
}

fn parse_macros(text: &str) -> Vec<(String, String)> {
    let mut macros = Vec::new();
    let mut pos = 0;
    while let Some(relative) = text[pos..].find("#macro") {
        let start = pos + relative;
        let header_start = start + "#macro".len();
        let header_end = text[header_start..]
            .find('\n')
            .map(|end| header_start + end)
            .unwrap_or(text.len());
        let header = text[header_start..header_end].trim();
        let Some(name) = header
            .split(|ch: char| ch == '(' || ch.is_whitespace())
            .find(|part| !part.is_empty())
            .map(str::to_string)
        else {
            pos = header_end;
            continue;
        };
        let Some(end_relative) = text[header_end..].find("#end") else {
            break;
        };
        let end = header_end + end_relative;
        macros.push((name, text[header_end..end].to_string()));
        pos = end + "#end".len();
    }
    macros
}

fn parse_macro_calls(text: &str) -> Vec<MacroCall> {
    let mut calls = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with("//")
            || line.starts_with("camera")
            || line.starts_with("global_settings")
        {
            continue;
        }
        let Some(open) = line.find('(') else {
            continue;
        };
        let name = line[..open].trim();
        if !(name.starts_with("modelprefix_") || name.starts_with("demprefix_")) {
            continue;
        }
        let Some(close) = find_matching_paren(line, open) else {
            continue;
        };
        let args = split_call_args(&line[open + 1..close]);
        if args.len() < 2 {
            continue;
        }
        let Some(pos) = parse_first_vec3(&args[0]) else {
            continue;
        };
        let Some(rot) = parse_first_vec3(&args[1]) else {
            continue;
        };
        let texture_arg = args
            .get(2)
            .map(|arg| arg.trim().trim_matches('"').to_string())
            .unwrap_or_default();
        calls.push(MacroCall {
            name: name.to_string(),
            pos,
            rot,
            texture_arg,
        });
    }
    calls
}

fn split_call_args(text: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut start = 0;
    let mut angle_depth = 0;
    let mut in_string = false;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        match ch {
            '"' => in_string = !in_string,
            '<' if !in_string => angle_depth += 1,
            '>' if !in_string && angle_depth > 0 => angle_depth -= 1,
            ',' if !in_string && angle_depth == 0 => {
                args.push(text[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < text.len() {
        args.push(text[start..].trim().to_string());
    }
    args
}

fn parse_camera(text: &str) -> Option<Camera> {
    let (open, close) = extract_named_braced(text, "camera")?;
    let block = &text[open + 1..close];

    let mut camera = Camera {
        angle: parse_float_after_keyword(block, "angle").unwrap_or(90.0),
        location: parse_vec3_after_keyword(block, "location").unwrap_or(Vec3::default()),
        look_at: parse_vec3_after_keyword(block, "look_at").unwrap_or(Vec3::new(1.0, 0.0, 0.0)),
        up: parse_vec3_after_keyword(block, "up").unwrap_or(Vec3::new(0.0, 0.0, 1.0)),
        right: parse_vec3_after_keyword(block, "right").unwrap_or(Vec3::new(1.0, 0.0, 0.0)),
        sky: parse_vec3_after_keyword(block, "sky").unwrap_or(Vec3::new(0.0, 0.0, 1.0)),
    };

    for transform in parse_camera_transforms(block) {
        match transform {
            CameraTransform::Rotate(rot) => {
                camera.location = rotate_vec(camera.location, rot);
                camera.look_at = rotate_vec(camera.look_at, rot);
                camera.up = rotate_vec(camera.up, rot);
                camera.right = rotate_vec(camera.right, rot);
                camera.sky = rotate_vec(camera.sky, rot);
            }
            CameraTransform::Translate(delta) => {
                camera.location += delta;
                camera.look_at += delta;
            }
        }
    }

    Some(camera)
}

#[derive(Clone, Copy, Debug)]
enum CameraTransform {
    Rotate(Vec3),
    Translate(Vec3),
}

fn parse_camera_transforms(block: &str) -> Vec<CameraTransform> {
    let mut transforms = Vec::new();
    let mut pos = 0;
    while pos < block.len() {
        let next_rotate = find_word(block, "rotate", pos);
        let next_translate = find_word(block, "translate", pos);
        match (next_rotate, next_translate) {
            (Some(r), Some(t)) if r < t => {
                if let Some(vec) = parse_first_vec3(&block[r..]) {
                    transforms.push(CameraTransform::Rotate(vec));
                }
                pos = r + "rotate".len();
            }
            (Some(r), None) => {
                if let Some(vec) = parse_first_vec3(&block[r..]) {
                    transforms.push(CameraTransform::Rotate(vec));
                }
                pos = r + "rotate".len();
            }
            (_, Some(t)) => {
                if let Some(vec) = parse_first_vec3(&block[t..]) {
                    transforms.push(CameraTransform::Translate(vec));
                }
                pos = t + "translate".len();
            }
            (None, None) => break,
        }
    }
    transforms
}

fn parse_lights(body: &str) -> Vec<LightTemplate> {
    let mut lights = Vec::new();
    let mut pos = 0;
    while let Some(relative) = find_word(body, "light_source", pos) {
        let start = relative;
        let Some(open) = body[start..].find('{').map(|index| start + index) else {
            break;
        };
        let Some(close) = find_matching_brace(body, open) else {
            break;
        };
        let block = &body[open + 1..close];
        if let Some(position) = parse_first_vec3(block) {
            let (color, intensity) = parse_light_color(block);
            lights.push(LightTemplate {
                position,
                color,
                intensity,
                fade_distance: parse_float_after_keyword(block, "fade_distance").unwrap_or(96.0),
                fade_power: parse_float_after_keyword(block, "fade_power").unwrap_or(1.5),
            });
        }
        pos = close + 1;
    }
    lights
}

fn parse_light_color(block: &str) -> (Color, f32) {
    let Some(pos) = block.find("rgb<") else {
        return (Color::splat(1.0), 1.0);
    };
    let Some(open) = block[pos..].find('<').map(|index| pos + index) else {
        return (Color::splat(1.0), 1.0);
    };
    let Some(close) = block[open..].find('>').map(|index| open + index) else {
        return (Color::splat(1.0), 1.0);
    };
    let values = parse_number_exprs(&block[open + 1..close]);
    let color = Color::new(
        *values.first().unwrap_or(&1.0),
        *values.get(1).unwrap_or(&1.0),
        *values.get(2).unwrap_or(&1.0),
    );

    let mut intensity = 1.0;
    let line_end = block[close + 1..]
        .find('\n')
        .map(|index| close + 1 + index)
        .unwrap_or(block.len());
    for part in block[close + 1..line_end].split('*').skip(1) {
        if let Some(value) = parse_leading_number_expr(part.trim()) {
            intensity *= value;
        }
    }

    (color, intensity)
}

fn parse_texture_list(block: &str) -> Vec<MaterialTemplate> {
    let mut materials = Vec::new();
    let mut pos = 0;
    while let Some(start) = find_word(block, "texture", pos) {
        let Some(open) = block[start..].find('{').map(|index| start + index) else {
            break;
        };
        let Some(close) = find_matching_brace(block, open) else {
            break;
        };
        materials.push(parse_texture_block(&block[open + 1..close]));
        pos = close + 1;
    }
    if materials.is_empty() {
        materials.push(MaterialTemplate::Key("default".to_string()));
    }
    materials
}

fn parse_texture_block(block: &str) -> MaterialTemplate {
    if let Some(color) = parse_pigment_color(block) {
        return MaterialTemplate::Color(color);
    }
    let comment = parse_texture_comment(block);
    if block.contains("png skin") {
        return MaterialTemplate::Texture {
            key: comment.unwrap_or_else(|| "skin".to_string()),
            source: TextureSource::Skin,
        };
    }
    if let Some(suffix) = parse_concat_texture_suffix(block) {
        return MaterialTemplate::Texture {
            key: comment.unwrap_or_else(|| suffix.clone()),
            source: TextureSource::TexturePrefixSuffix(suffix),
        };
    }

    if let Some(path) = find_first_quoted_after(block, "png") {
        return MaterialTemplate::Texture {
            key: comment.unwrap_or_else(|| path.clone()),
            source: TextureSource::File(path),
        };
    }

    if let Some(comment) = comment {
        return MaterialTemplate::Key(comment);
    }

    MaterialTemplate::Key("default".to_string())
}

fn parse_texture_comment(block: &str) -> Option<String> {
    block
        .find("//")
        .map(|comment_start| {
            let rest = &block[comment_start + 2..];
            rest.lines().next().unwrap_or("").trim().to_string()
        })
        .filter(|comment| !comment.is_empty())
}

fn parse_concat_texture_suffix(block: &str) -> Option<String> {
    let concat = find_word(block, "concat", 0)?;
    find_first_quoted_after(&block[concat..], "concat")
}

fn parse_pigment_color(block: &str) -> Option<Color> {
    if let Some(pos) = block.find("rgbf<") {
        return parse_color_angle(&block[pos + "rgbf".len()..]);
    }
    if let Some(pos) = block.find("rgb<") {
        return parse_color_angle(&block[pos + "rgb".len()..]);
    }
    if block.contains("rgb 1") {
        return Some(Color::splat(1.0));
    }
    None
}

fn parse_color_angle(text: &str) -> Option<Color> {
    let open = text.find('<')?;
    let close = text[open + 1..].find('>')? + open + 1;
    let values = parse_number_exprs(&text[open + 1..close]);
    Some(Color::new(
        *values.first().unwrap_or(&1.0),
        *values.get(1).unwrap_or(&1.0),
        *values.get(2).unwrap_or(&1.0),
    ))
}

fn find_first_quoted_after(text: &str, keyword: &str) -> Option<String> {
    let keyword_pos = text.find(keyword)?;
    let first_quote = text[keyword_pos..].find('"')? + keyword_pos;
    let second_quote = text[first_quote + 1..].find('"')? + first_quote + 1;
    Some(text[first_quote + 1..second_quote].to_string())
}

fn find_mesh2_blocks(body: &str) -> Vec<&str> {
    let mut blocks = Vec::new();
    let mut pos = 0;
    while let Some(start) = find_word(body, "mesh2", pos) {
        let Some(open) = body[start..].find('{').map(|index| start + index) else {
            break;
        };
        let Some(close) = find_matching_brace(body, open) else {
            break;
        };
        blocks.push(&body[start..=close]);
        pos = close + 1;
    }
    blocks
}

fn extract_named_braced(text: &str, keyword: &str) -> Option<(usize, usize)> {
    let start = find_word(text, keyword, 0)?;
    let open = text[start..].find('{')? + start;
    let close = find_matching_brace(text, open)?;
    Some((open, close))
}

fn find_matching_paren(text: &str, open: usize) -> Option<usize> {
    find_matching_delimited(text, open, b'(', b')')
}

fn find_matching_brace(text: &str, open: usize) -> Option<usize> {
    find_matching_delimited(text, open, b'{', b'}')
}

fn find_matching_delimited(text: &str, open: usize, left: u8, right: u8) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(open).copied()? != left {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut i = open;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte == b'"' {
            in_string = !in_string;
        } else if !in_string {
            if byte == left {
                depth += 1;
            } else if byte == right {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
        }
        i += 1;
    }
    None
}

fn find_word(text: &str, needle: &str, start: usize) -> Option<usize> {
    let mut pos = start;
    while let Some(relative) = text[pos..].find(needle) {
        let index = pos + relative;
        let before = index
            .checked_sub(1)
            .and_then(|i| text.as_bytes().get(i).copied())
            .map(is_ident_byte)
            .unwrap_or(false);
        let after = text
            .as_bytes()
            .get(index + needle.len())
            .copied()
            .map(is_ident_byte)
            .unwrap_or(false);
        if !before && !after {
            return Some(index);
        }
        pos = index + needle.len();
    }
    None
}

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn parse_vec3_list(text: &str) -> Vec<Vec3> {
    let mut vectors = Vec::new();
    let mut pos = 0;
    while let Some(open_relative) = text[pos..].find('<') {
        let open = pos + open_relative;
        let Some(close) = text[open..].find('>').map(|index| open + index) else {
            break;
        };
        if let Some(vector) = parse_angle_vec3(&text[open + 1..close]) {
            vectors.push(vector);
        }
        pos = close + 1;
    }
    vectors
}

fn parse_vec2_list(text: &str) -> Vec<Vec2> {
    let mut vectors = Vec::new();
    let mut pos = 0;
    while let Some(open_relative) = text[pos..].find('<') {
        let open = pos + open_relative;
        let Some(close) = text[open..].find('>').map(|index| open + index) else {
            break;
        };
        if let Some(vector) = parse_angle_vec2(&text[open + 1..close]) {
            vectors.push(vector);
        }
        pos = close + 1;
    }
    vectors
}

fn parse_face_indices(text: &str) -> Vec<FaceTemplate> {
    let mut faces = Vec::new();
    let mut pos = 0;
    while let Some(open_relative) = text[pos..].find('<') {
        let open = pos + open_relative;
        let Some(close) = text[open..].find('>').map(|index| open + index) else {
            break;
        };
        let indices: Vec<usize> = text[open + 1..close]
            .split(',')
            .filter_map(|part| part.trim().parse::<usize>().ok())
            .collect();
        if indices.len() >= 3 {
            let material_index = parse_material_index_after(&text[close + 1..]).unwrap_or(0);
            faces.push(FaceTemplate {
                indices: [indices[0], indices[1], indices[2]],
                material_index,
            });
        }
        pos = close + 1;
    }
    faces
}

fn parse_index_triples(text: &str) -> Vec<[usize; 3]> {
    let mut triples = Vec::new();
    let mut pos = 0;
    while let Some(open_relative) = text[pos..].find('<') {
        let open = pos + open_relative;
        let Some(close) = text[open..].find('>').map(|index| open + index) else {
            break;
        };
        let indices: Vec<usize> = text[open + 1..close]
            .split(',')
            .filter_map(|part| part.trim().parse::<usize>().ok())
            .collect();
        if indices.len() >= 3 {
            triples.push([indices[0], indices[1], indices[2]]);
        }
        pos = close + 1;
    }
    triples
}

fn uv_triangle(uvs: &[Vec2], indices: [usize; 3]) -> Option<[Vec2; 3]> {
    Some([
        *uvs.get(indices[0])?,
        *uvs.get(indices[1])?,
        *uvs.get(indices[2])?,
    ])
}

fn parse_material_index_after(text: &str) -> Option<usize> {
    let text = text.trim_start();
    let text = text.strip_prefix(',')?.trim_start();
    let mut end = 0;
    for (index, ch) in text.char_indices() {
        if ch.is_ascii_digit() {
            end = index + ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        None
    } else {
        text[..end].parse().ok()
    }
}

fn parse_first_vec3(text: &str) -> Option<Vec3> {
    let open = text.find('<')?;
    let close = text[open..].find('>')? + open;
    parse_angle_vec3(&text[open + 1..close])
}

fn parse_angle_vec3(text: &str) -> Option<Vec3> {
    let values = parse_number_exprs(text);
    if values.len() < 3 {
        None
    } else {
        Some(Vec3::new(values[0], values[1], values[2]))
    }
}

fn parse_angle_vec2(text: &str) -> Option<Vec2> {
    let values = parse_number_exprs(text);
    if values.len() < 2 {
        None
    } else {
        Some(Vec2::new(values[0], values[1]))
    }
}

fn parse_number_exprs(text: &str) -> Vec<f32> {
    text.split(',')
        .filter_map(|part| eval_number_expr(part.trim()))
        .collect()
}

fn parse_float_after_keyword(text: &str, keyword: &str) -> Option<f32> {
    let start = find_word(text, keyword, 0)? + keyword.len();
    parse_leading_number_expr(text[start..].trim_start())
}

fn parse_vec3_after_keyword(text: &str, keyword: &str) -> Option<Vec3> {
    let start = find_word(text, keyword, 0)? + keyword.len();
    parse_first_vec3(&text[start..])
}

fn parse_leading_number_expr(text: &str) -> Option<f32> {
    let mut end = 0;
    for (index, ch) in text.char_indices() {
        if ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E' | '/') {
            end = index + ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        None
    } else {
        eval_number_expr(&text[..end])
    }
}

fn eval_number_expr(text: &str) -> Option<f32> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let mut parts = text.split('/');
    let mut value = parts.next()?.trim().parse::<f32>().ok()?;
    for part in parts {
        value /= part.trim().parse::<f32>().ok()?;
    }
    Some(value)
}

fn transform_point(point: Vec3, rot: Vec3, translate: Vec3) -> Vec3 {
    rotate_vec(point, rot) + translate
}

fn rotate_vec(vector: Vec3, rot: Vec3) -> Vec3 {
    let mut v = vector;
    if rot.x != 0.0 {
        let (sin, cos) = (rot.x * PI / 180.0).sin_cos();
        v = Vec3::new(v.x, v.y * cos - v.z * sin, v.y * sin + v.z * cos);
    }
    if rot.y != 0.0 {
        let (sin, cos) = (rot.y * PI / 180.0).sin_cos();
        v = Vec3::new(v.x * cos + v.z * sin, v.y, -v.x * sin + v.z * cos);
    }
    if rot.z != 0.0 {
        let (sin, cos) = (rot.z * PI / 180.0).sin_cos();
        v = Vec3::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos, v.z);
    }
    v
}

fn material_color_from_key(key: &str) -> Color {
    let lower = key.to_ascii_lowercase();
    if lower.contains("slime") {
        return Color::new(0.14, 0.56, 0.08);
    }
    if lower.contains("water") || lower.contains("tele") || lower.contains("sky") {
        return Color::new(0.16, 0.28, 0.52);
    }
    if lower.contains("lava") || lower.contains("fire") {
        return Color::new(0.90, 0.30, 0.08);
    }
    if lower.contains("light") || lower.contains("tlight") {
        return Color::new(0.95, 0.78, 0.35);
    }
    if lower.contains("door") || lower.contains("wood") {
        return Color::new(0.45, 0.30, 0.18);
    }
    if lower.contains("floor") || lower.contains("ground") {
        return Color::new(0.42, 0.37, 0.30);
    }
    if lower.contains("wall") || lower.contains("tech") || lower.contains("comp") {
        return Color::new(0.38, 0.42, 0.43);
    }
    if lower.contains("skin") || lower.contains(".mdl") || lower.contains("progs") {
        let h = hash32(key);
        let warm = ((h >> 8) & 0xff) as f32 / 255.0;
        return Color::new(0.30 + warm * 0.25, 0.24 + warm * 0.18, 0.18 + warm * 0.10);
    }

    let h = hash32(key);
    let r = 0.24 + ((h & 0xff) as f32 / 255.0) * 0.34;
    let g = 0.24 + (((h >> 8) & 0xff) as f32 / 255.0) * 0.34;
    let b = 0.24 + (((h >> 16) & 0xff) as f32 / 255.0) * 0.34;
    Color::new(r, g, b)
}

fn hash32(text: &str) -> u32 {
    let mut hash = 2_166_136_261u32;
    for byte in text.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vector_exprs() {
        assert_eq!(parse_first_vec3("<6/256,74/256,0>").unwrap().x, 6.0 / 256.0);
        let v = parse_first_vec3("<1,-2.5,3.5e-2>").unwrap();
        assert!((v.y + 2.5).abs() < 1.0e-6);
        assert!((v.z - 0.035).abs() < 1.0e-6);
    }

    #[test]
    fn splits_macro_call_args() {
        let args = split_call_args(r#"<1,2,3>,<4,5,6>,"maps/e1m1.bsp""#);
        assert_eq!(args.len(), 3);
        assert_eq!(args[2], r#""maps/e1m1.bsp""#);
    }

    #[test]
    fn parses_face_materials() {
        let faces = parse_face_indices("2, <0,1,2>,4,<2,3,0>,7");
        assert_eq!(faces.len(), 2);
        assert_eq!(faces[0].indices, [0, 1, 2]);
        assert_eq!(faces[0].material_index, 4);
        assert_eq!(faces[1].material_index, 7);
    }

    #[test]
    fn parses_uv_vectors_and_indices() {
        let uvs = parse_vec2_list("3, <0,0>,<1.5,-2>,<6/256,74/256>");
        assert_eq!(uvs.len(), 3);
        assert!((uvs[2].x - 6.0 / 256.0).abs() < 1.0e-6);
        assert!((uvs[2].y - 74.0 / 256.0).abs() < 1.0e-6);

        let indices = parse_index_triples("2, <0,1,2>,<2,1,0>");
        assert_eq!(indices, vec![[0, 1, 2], [2, 1, 0]]);
    }

    #[test]
    fn parses_texture_sources() {
        let material = parse_texture_block(
            r#"// tech04_3 (#6)
            pigment {
              image_map {
                png concat(textureprefix, "/texture_6.png")
              }
            }"#,
        );
        assert_eq!(
            material
                .texture_path("maps/e1m1.bsp")
                .unwrap()
                .to_string_lossy(),
            "maps/e1m1.bsp/texture_6.png"
        );

        let skin = parse_texture_block("pigment { image_map { png skin } }");
        assert_eq!(
            skin.texture_path("progs/soldier.mdl/skin_0.png")
                .unwrap()
                .to_string_lossy(),
            "progs/soldier.mdl/skin_0.png"
        );
    }

    #[test]
    fn texture_sampler_uses_png_row_order_for_v() {
        let texture = Texture {
            width: 1,
            height: 2,
            pixels: vec![Color::new(1.0, 0.0, 0.0), Color::new(0.0, 0.0, 1.0)],
        };

        let top = texture.sample(Vec2::new(0.5, 0.25));
        let bottom = texture.sample(Vec2::new(0.5, 0.75));

        assert!(top.r > 0.99);
        assert!(top.b < 0.01);
        assert!(bottom.r < 0.01);
        assert!(bottom.b > 0.99);
    }

    #[test]
    fn texture_decoder_converts_srgb_to_linear() {
        let pixels = decode_texture_pixels(png::ColorType::Rgb, &[128, 255, 0]).unwrap();
        assert_eq!(pixels.len(), 1);
        assert!((pixels[0].r - 0.21586).abs() < 1.0e-4);
        assert!((pixels[0].g - 1.0).abs() < 1.0e-6);
        assert!((pixels[0].b - 0.0).abs() < 1.0e-6);
    }

    #[test]
    fn adaptive_aa_triggers_only_on_contrast() {
        let uniform = vec![Color::splat(0.25); 9];
        assert!(!should_antialias(&uniform, 1, 1, 3, 3, 0.12));

        let mut edge = uniform;
        edge[1] = Color::splat(0.9);
        assert!(should_antialias(&edge, 1, 1, 3, 3, 0.12));
    }

    #[test]
    fn qdqr_camera_basis_is_perpendicular_after_look_at() {
        let camera = parse_camera(
            "camera {
              angle 100
              location <0,0,0>
              sky <0,0,1>
              up <0,0,9>
              right <-16,0,0>
              look_at <1,0,0>
              rotate <0,0,90>
              translate <0,0,10>
            }",
        )
        .unwrap();
        let (forward, right, up) = camera.basis();

        assert!(forward.dot(right).abs() < 1.0e-5);
        assert!(forward.dot(up).abs() < 1.0e-5);
        assert!(right.dot(up).abs() < 1.0e-5);
        assert!(forward.y > 0.99);
        assert!(right.x > 0.99);
        assert!(up.z > 0.99);
    }

    #[test]
    fn aabb_accepts_boundary_touch_hits() {
        let bbox = Aabb {
            min: Vec3::new(1.0, 1.0, 1.0),
            max: Vec3::new(2.0, 2.0, 2.0),
        };
        let ray = Ray {
            origin: Vec3::new(0.0, 0.0, 2.0),
            dir: Vec3::new(1.0, 1.0, -1.0).normalize_or(Vec3::default()),
        };

        assert!(bbox.hit(ray, f32::INFINITY));
    }

    #[test]
    fn scene_occlusion_stops_on_any_blocking_triangle() {
        let triangle = Triangle::new(
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Color::splat(1.0),
            None,
            None,
            None,
        )
        .unwrap();
        let scene = Scene::new(
            Camera {
                angle: 90.0,
                location: Vec3::new(0.0, 0.0, 2.0),
                look_at: Vec3::default(),
                up: Vec3::new(0.0, 1.0, 0.0),
                right: Vec3::new(1.0, 0.0, 0.0),
                sky: Vec3::new(0.0, 1.0, 0.0),
            },
            vec![triangle],
            Vec::new(),
            Vec::new(),
            false,
        );
        let ray = Ray {
            origin: Vec3::new(0.0, 0.0, 1.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
        };

        assert!(scene.occluded(ray, f32::INFINITY));
        assert!(!scene.occluded(ray, 0.5));
    }

    #[test]
    fn nearest_light_iteration_preserves_replacement_slot_order() {
        let lights: Vec<Light> = [5.0, 4.0, 3.0, 2.0, 1.0]
            .into_iter()
            .map(|x| Light {
                position: Vec3::new(x, 0.0, 0.0),
                color: Color::splat(1.0),
                intensity: 1.0,
                fade_distance: 1.0,
                fade_power: 1.0,
            })
            .collect();
        let point = Vec3::default();

        let mut limited = Vec::new();
        for_nearest_lights(point, &lights, 3, |index| limited.push(index));
        assert_eq!(limited, vec![3, 4, 2]);

        let mut unlimited = Vec::new();
        for_nearest_lights(point, &lights, 10, |index| unlimited.push(index));
        assert_eq!(unlimited, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn vertex_normals_average_connected_face_normals() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let faces = vec![
            FaceTemplate {
                indices: [0, 1, 2],
                material_index: 0,
            },
            FaceTemplate {
                indices: [0, 3, 1],
                material_index: 0,
            },
        ];

        let normals = compute_vertex_normals(&vertices, &faces);

        assert!(normals[0].x.abs() < 1.0e-6);
        assert!((normals[0].y - 0.70710677).abs() < 1.0e-5);
        assert!((normals[0].z - 0.70710677).abs() < 1.0e-5);
        assert_eq!(normals[2].z, 1.0);
    }

    #[test]
    fn smooth_normals_apply_only_to_mdl_calls_when_enabled() {
        let mesh = MeshTemplate {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            uvs: Vec::new(),
            faces: vec![
                FaceTemplate {
                    indices: [0, 1, 2],
                    material_index: 0,
                },
                FaceTemplate {
                    indices: [0, 3, 1],
                    material_index: 0,
                },
            ],
            uv_indices: Vec::new(),
            materials: vec![MaterialTemplate::Color(Color::splat(1.0))],
        };
        let mdl_call = MacroCall {
            name: "demprefix_progs_test_mdl_0".to_string(),
            pos: Vec3::default(),
            rot: Vec3::default(),
            texture_arg: "progs/test.mdl/skin_0.png".to_string(),
        };
        let bsp_call = MacroCall {
            name: "modelprefix_maps_test_bsp_0".to_string(),
            pos: Vec3::default(),
            rot: Vec3::default(),
            texture_arg: "maps/test.bsp".to_string(),
        };

        let mut mdl_triangles = Vec::new();
        let mut no_textures = None;
        mesh.instantiate(&mdl_call, &mut mdl_triangles, &mut no_textures, true);

        let mut bsp_triangles = Vec::new();
        let mut no_textures = None;
        mesh.instantiate(&bsp_call, &mut bsp_triangles, &mut no_textures, true);

        let mut disabled_triangles = Vec::new();
        let mut no_textures = None;
        mesh.instantiate(&mdl_call, &mut disabled_triangles, &mut no_textures, false);

        assert!((mdl_triangles[0].vertex_normals[0].y - 0.70710677).abs() < 1.0e-5);
        assert!((mdl_triangles[0].vertex_normals[0].z - 0.70710677).abs() < 1.0e-5);
        assert!(bsp_triangles[0].vertex_normals[0].y.abs() < 1.0e-6);
        assert!((bsp_triangles[0].vertex_normals[0].z - 1.0).abs() < 1.0e-6);
        assert!(disabled_triangles[0].vertex_normals[0].y.abs() < 1.0e-6);
        assert!((disabled_triangles[0].vertex_normals[0].z - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn intersects_triangle() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Color::splat(1.0),
            None,
            None,
            None,
        )
        .unwrap();
        let ray = Ray {
            origin: Vec3::new(0.25, 0.25, 1.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
        };
        assert!(tri.intersect(ray, f32::INFINITY).is_some());
    }
}
