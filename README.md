# qray

vibe coded raytracer for [qpov](https://github.com/ThomasHabets/qpov) Quake
demos.

In the past I've rendered these with povray. And it can render [really pretty
images][pretty], with the only downside that it's really slow.

qray is me experimenting with vibe coding some renderer that could work better
for this specific purpose.

## Turning the frames into video

```
ffmpeg -framerate 30 \
  -i 'renders/frame-%08d.png' \
  -c:v ffv1 \
  -level 3 \
  -pix_fmt rgb24 \
  output.mkv
```

[pretty]: https://cement.retrofitta.se/tmp/frame-738-povray.png
