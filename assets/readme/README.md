# README teaser

`teaser.gif` shows the wiping, insertion, and pipette-transfer demonstrations
from the [ViTaL project video](https://yilin-wu98.github.io/vital_website/static/videos/teaser_web.mp4).
It uses the 02:10–02:26 excerpt, without changing the playback speed.

The GIF is 960 × 540 pixels, 10 frames per second, and loops continuously.
The original video remains available on the
[project website](https://yilin-wu98.github.io/vital_website/).

To regenerate it with FFmpeg after downloading the original video:

```bash
ffmpeg -ss 130 -i teaser_web.mp4 -t 16 \
  -filter_complex 'fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle' \
  -loop 0 assets/readme/teaser.gif
```
