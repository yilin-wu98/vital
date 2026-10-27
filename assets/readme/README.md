# README teaser

`teaser.gif` shows the method overview, including world-model prediction,
visual mode selection, and tactile refinement,
from the [ViTaL project video](https://yilin-wu98.github.io/vital_website/static/videos/teaser_web.mp4).
It uses the full 00:00–02:08 method-overview section, without changing the
playback speed.

The GIF is 960 × 540 pixels, 8 frames per second, and loops continuously.
Clicking it in the main README opens the original overview video.

To regenerate it with FFmpeg after downloading the original video:

```bash
ffmpeg -i teaser_web.mp4 -t 128 \
  -filter_complex 'fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle' \
  -loop 0 assets/readme/teaser.gif
```
