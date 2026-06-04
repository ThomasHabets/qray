use clap::Parser;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

type Result<T> = std::result::Result<T, String>;

fn main() {
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
#[command(
    name = "qray",
    about = "Render qdqr POV frame files to binary PPM images"
)]
struct Config {
    #[arg(
        value_name = "FRAME_OR_DIR",
        default_value = "qdqr-e1m1-20160502/frame-00000738.pov"
    )]
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
        out.set_extension("ppm");
        out
    });

    let mut library = PovLibrary::default();
    render_frame(&mut library, &config.input, &output, config)
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
    for frame in frames {
        let mut output = output_dir.join(
            frame
                .file_stem()
                .ok_or_else(|| format!("invalid frame path `{}`", frame.display()))?,
        );
        output.set_extension("ppm");
        render_frame(&mut library, &frame, &output, config)?;
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

fn render_frame(
    library: &mut PovLibrary,
    input: &Path,
    output: &Path,
    config: &Config,
) -> Result<()> {
    let started = Instant::now();
    let build = build_scene(library, input, config)?;
    if config.stats {
        eprintln!(
            "{}: {} calls, {} triangles, {} lights, {} warnings, parse/build {:.2}s",
            input.display(),
            build.call_count,
            build.scene.triangles.len(),
            build.scene.lights.len(),
            build.warnings.len(),
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
    write_ppm(output, config.width, config.height, &pixels)?;
    if config.stats {
        eprintln!(
            "{} -> {} in {:.2}s",
            input.display(),
            output.display(),
            render_started.elapsed().as_secs_f32()
        );
    }

    Ok(())
}

fn build_scene(library: &mut PovLibrary, input: &Path, config: &Config) -> Result<SceneBuild> {
    let text = fs::read_to_string(input)
        .map_err(|err| format!("failed to read frame `{}`: {err}", input.display()))?;
    let frame_dir = input
        .parent()
        .ok_or_else(|| format!("frame `{}` has no parent directory", input.display()))?;

    let static_lights = library.load_includes(frame_dir, &text)?;

    let mut warnings = std::mem::take(&mut library.warnings);
    let camera =
        parse_camera(&text).ok_or_else(|| format!("missing camera in `{}`", input.display()))?;
    let calls = parse_macro_calls(&text);

    let mut triangles = Vec::new();
    let mut lights = Vec::new();
    for call in &calls {
        if let Some(template) = library.macros.get(&call.name) {
            template.instantiate(call, &mut triangles, &mut lights);
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

    let scene = Scene::new(camera, triangles, lights, config.stats);
    Ok(SceneBuild {
        scene,
        call_count: calls.len(),
        warnings,
    })
}

struct SceneBuild {
    scene: Scene,
    call_count: usize,
    warnings: Vec<String>,
}

#[derive(Default)]
struct PovLibrary {
    macros: HashMap<String, MacroTemplate>,
    parsed_files: HashSet<PathBuf>,
    file_lights: HashMap<PathBuf, Vec<LightTemplate>>,
    warnings: Vec<String>,
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
    ) {
        for mesh in &self.meshes {
            mesh.instantiate(call, triangles);
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
    faces: Vec<FaceTemplate>,
    materials: Vec<MaterialTemplate>,
}

impl MeshTemplate {
    fn instantiate(&self, call: &MacroCall, triangles: &mut Vec<Triangle>) {
        if self.vertices.is_empty() || self.faces.is_empty() {
            return;
        }

        let transformed: Vec<Vec3> = self
            .vertices
            .iter()
            .map(|&vertex| transform_point(vertex, call.rot, call.pos))
            .collect();

        for face in &self.faces {
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
            if let Some(triangle) = Triangle::new(v0, v1, v2, color) {
                triangles.push(triangle);
            }
        }
    }
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
    Skin,
}

impl MaterialTemplate {
    fn resolve(&self, texture_arg: &str) -> Color {
        match self {
            Self::Color(color) => *color,
            Self::Key(key) => material_color_from_key(key),
            Self::Skin => material_color_from_key(texture_arg),
        }
    }
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
        let (forward, right, up) = self.basis();

        let aspect = if self.right.length() > 1.0e-8 && self.up.length() > 1.0e-8 {
            self.right.length() / self.up.length()
        } else {
            width as f32 / height as f32
        };
        let angle = self.angle.clamp(1.0, 175.0) * PI / 180.0;
        let viewport_width = 2.0 * (angle * 0.5).tan();
        let viewport_height = viewport_width / aspect;
        let px = ((x as f32 + 0.5) / width as f32 - 0.5) * viewport_width;
        let py = (0.5 - (y as f32 + 0.5) / height as f32) * viewport_height;

        Ray {
            origin: self.location,
            dir: (forward + right * px + up * py).normalize_or(forward),
        }
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
    bvh: BvhNode,
}

impl Scene {
    fn new(camera: Camera, triangles: Vec<Triangle>, lights: Vec<Light>, stats: bool) -> Self {
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
        self.hit(ray, max_t).is_some()
    }
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    v0: Vec3,
    e1: Vec3,
    e2: Vec3,
    normal: Vec3,
    color: Color,
    bbox: Aabb,
    centroid: Vec3,
}

impl Triangle {
    fn new(v0: Vec3, v1: Vec3, v2: Vec3, color: Color) -> Option<Self> {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let normal = e1.cross(e2).normalize_or(Vec3::default());
        if normal.length() < 1.0e-8 {
            return None;
        }
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
            color,
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

    fn hit(self, ray: Ray, max_t: f32) -> bool {
        let mut t_min: f32 = 0.0;
        let mut t_max: f32 = max_t;

        for axis in 0..3 {
            let origin = ray.origin.component(axis);
            let dir = ray.dir.component(axis);
            let min = self.min.component(axis);
            let max = self.max.component(axis);
            if dir.abs() < 1.0e-8 {
                if origin < min || origin > max {
                    return false;
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
            if t_max <= t_min {
                return false;
            }
        }

        true
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
        match self {
            Self::Leaf {
                bbox,
                triangles: leaf,
            } => {
                if !bbox.hit(ray, best.t) {
                    return;
                }
                for &index in leaf {
                    if let Some((t, u, v)) = triangles[index].intersect(ray, best.t) {
                        best.t = t;
                        best.triangle_index = index;
                        best.bary_u = u;
                        best.bary_v = v;
                    }
                }
            }
            Self::Branch { bbox, left, right } => {
                if !bbox.hit(ray, best.t) {
                    return;
                }
                left.hit(ray, triangles, best);
                right.hit(ray, triangles, best);
            }
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
    pixels
}

fn trace(scene: &Scene, ray: Ray, config: &Config) -> Color {
    let Some(hit) = scene.hit(ray, f32::INFINITY) else {
        return background_color(ray.dir);
    };

    let triangle = scene.triangles[hit.triangle_index];
    let point = ray.origin + ray.dir * hit.t;
    let mut normal = triangle.normal;
    if normal.dot(ray.dir) > 0.0 {
        normal = -normal;
    }

    let mut color = triangle.color * 0.14;
    color += triangle.color * (0.08 * normal.z.max(0.0));
    color += triangle.color * (0.05 * (-ray.dir).dot(normal).max(0.0));

    for index in nearest_lights(point, &scene.lights, config.max_lights) {
        let light = scene.lights[index];
        let to_light = light.position - point;
        let distance = to_light.length();
        if distance < 1.0e-4 {
            continue;
        }
        let light_dir = to_light / distance;
        let ndotl = normal.dot(light_dir).max(0.0);
        if ndotl <= 0.0 {
            continue;
        }

        if config.shadows {
            let shadow_ray = Ray {
                origin: point + normal * 0.03,
                dir: light_dir,
            };
            if scene.occluded(shadow_ray, distance - 0.06) {
                continue;
            }
        }

        let attenuation = light_attenuation(distance, light.fade_distance, light.fade_power);
        let strength = ndotl * light.intensity * attenuation;
        color += triangle.color * light.color * strength;
    }

    color.tone_map().clamp01()
}

fn nearest_lights(point: Vec3, lights: &[Light], limit: usize) -> Vec<usize> {
    if limit == 0 || lights.is_empty() {
        return Vec::new();
    }
    if lights.len() <= limit {
        return (0..lights.len()).collect();
    }

    let mut nearest: Vec<(f32, usize)> = Vec::with_capacity(limit);
    for (index, light) in lights.iter().enumerate() {
        let distance2 = (light.position - point).dot(light.position - point);
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

    nearest.into_iter().map(|(_, index)| index).collect()
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

fn write_ppm(path: &Path, width: usize, height: usize, pixels: &[Color]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create `{}`: {err}", parent.display()))?;
        }
    }
    let file = File::create(path)
        .map_err(|err| format!("failed to create output `{}`: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(format!("P6\n{width} {height}\n255\n").as_bytes())
        .map_err(|err| format!("failed to write `{}`: {err}", path.display()))?;
    for color in pixels {
        writer
            .write_all(&[
                to_ppm_byte(color.r),
                to_ppm_byte(color.g),
                to_ppm_byte(color.b),
            ])
            .map_err(|err| format!("failed to write `{}`: {err}", path.display()))?;
    }
    Ok(())
}

fn to_ppm_byte(value: f32) -> u8 {
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

    let face_block = extract_named_braced(mesh, "face_indices")?;
    let faces = parse_face_indices(&mesh[face_block.0 + 1..face_block.1]);

    let materials = extract_named_braced(mesh, "texture_list")
        .map(|(open, close)| parse_texture_list(&mesh[open + 1..close]))
        .unwrap_or_else(|| vec![MaterialTemplate::Key("default".to_string())]);

    Some(MeshTemplate {
        vertices,
        faces,
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
    if block.contains("png skin") {
        return MaterialTemplate::Skin;
    }

    let comment = block.find("//").map(|comment_start| {
        let rest = &block[comment_start + 2..];
        rest.lines().next().unwrap_or("").trim().to_string()
    });
    if let Some(comment) = comment.filter(|comment| !comment.is_empty()) {
        return MaterialTemplate::Key(comment);
    }

    if let Some(path) = find_first_quoted_after(block, "png") {
        return MaterialTemplate::Key(path);
    }

    MaterialTemplate::Key("default".to_string())
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
    fn intersects_triangle() {
        let tri = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Color::splat(1.0),
        )
        .unwrap();
        let ray = Ray {
            origin: Vec3::new(0.25, 0.25, 1.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
        };
        assert!(tri.intersect(ray, f32::INFINITY).is_some());
    }
}
