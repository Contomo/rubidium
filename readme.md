# Rubidium setup & config reference

Rubidium is a Klipper extra that generates a line-pattern (currently only 'lines'), prints it, scans it while recording video, and optionally analyses the scan video.

## Installation

### Install script (recommended)

Run the repository’s installer (symlinks into Klipper’s extras):

```bash
wget -O - https://raw.githubusercontent.com/Contomo/rubidium/main/install.sh | bash
```

### Manual install

1. Clone the repo into `~/rubidium`
2. Symlink `~/rubidium/rubidium` into `~/klipper/klippy/extras/rubidium`
3. Install requirements
    ```bash
    sudo apt-get update
    sudo apt-get install -y ffmpeg
    ~/klippy-env/bin/pip install -U numpy opencv-python matplotlib
    ```
4. Restart Klipper

## Dependencies

Rubidium uses:
- `ffmpeg` for recording and cutting video
- `numpy` + `opencv-python` (`cv2`) for analysis
- `matplotlib` for plot rendering (analysis_write_plots)


## Configuration layout

Rubidium sections are:

- `[rubidium]` – base/shared defaults (used as fallback by all other rubidium sections)
- `[rubidium pattern]` / `[rubidium pattern <name>]` – pattern definitions
- `[rubidium print]` – registers `RUBIDIUM_PRINT`
- `[rubidium scan]` – registers `RUBIDIUM_SCAN`
- `[rubidium analyse]` – registers `RUBIDIUM_ANALYSE`
- `[rubidium latency]` – registers `RUBIDIUM_LATENCY`

*Only the unnamed `print/scan/analyse/latency` sections are supported right now.*

## Commands overview

### `RUBIDIUM_PRINT`
Print the selected pattern (streams gcode via virtual_sdcard).

Common parameters:
- `PATTERN=<name>` – choose a pattern (defaults to `[rubidium pattern]`)
- `TRAVEL_SPEED=<mm/s>` – override travel speed for this run
- `FLOW=<mult>`
- `LINE_WIDTH=<mm>` or `LINE_WIDTH_PCT=<pct>`
- `LAYER_HEIGHT=<mm>`
- `PRINT_SPEED=<mm/s>` – brim speed
- `BRIM_WALLS=<int>`
- `BRIM_OVERLAP=<pct>`
- `SPEED_<LABEL>=<mm/s>` – override an entry in `speeds:` for this run

> you can place the settings for patterns also into this section as the "default" pattern.  
> you need to configure a "this name this speed" (`speeds`) section in here.

### `RUBIDIUM_SCAN`
Traverse the pattern lines (G1) and record video + per-line clips.

Common parameters:
- `PATTERN=<name>`
- `TRAVEL_SPEED=<mm/s>`
- `SCAN_SPEED=<mm/s>`
- `SCAN_BUFFER=<mm>`
- `VIDEO_LATENCY_MS=<ms>` – per-run latency compensation (subtracted from timestamps)

### `RUBIDIUM_ANALYSE`
Analyse a scan session directory.

Parameters:
- `SESSION=<name>` – session folder name (inside scan_dir), or
- `DIR=<path>` – explicit directory

If neither is provided, Rubidium will use the most recent scan directory under `scan_dir`.

### `RUBIDIUM_LATENCY`
Measures camera-to-motion latency by arming at a lookahead-timed moment and triggering on frame differences.

Useful parameters:
- `AXIS=X|Y` (default depends on code path)
- `DIST=<mm>`
- `SPEED=<mm/s>`
- `TIMEOUT=<s>`
- `RETURN=1` (return to start position)

> *Note that this does not actually tell you the value to input into scans `video_latency_ms`*

## Video capture model

Scan mode (`[rubidium scan]`) uses `VideoInput`:

- A “dump” recording is written into a new folder:
  `scan_dir/recording_<YYYY-MM-DD_HH-MM>_<pid>/`
- For each printed/scanned line, Rubidium logs `start/end` marks at motion-planner time.
- Each end mark queues a cut job, producing:
  `rubidium_cut_000.mp4`, `rubidium_cut_001.mp4`, …
- A JSON log is written alongside the dump and clips.

Key settings to edit (inside `[rubidium scan]`):
- `video_source` – needs to be provided, pointing at a device or url *(`crowsnest` stream for example)*
- `video_latency_ms` – shifts marks earlier to compensate camera latency
- `video_extra_input_args` / `video_extra_output_args` – control ffmpeg recording
- `video_cut_extra_input_args` / `video_cut_extra_output_args` – control ffmpeg cuts

## Template context

The per-mode templates (`start_gcode`, `before_line_gcode`, `after_line_gcode`, `end_gcode`) are rendered with a context that includes:
- `params` passed into the respective gcode command
- `mode` – `"print"` or `"scan"`
- `travel_speed`, `scan_speed`, `scan_buffer`
- `pattern` – selected pattern name
- `pattern_settings` – dict of resolved pattern settings
- `pattern_segments` – list of `(tag, fraction)` pairs
- `speeds` – dict mapping speed labels to mm/s
- `line` – current `Line` object (for `before_line_gcode` / `after_line_gcode`)
  - `line.idx`
  - `line.parameter_value`
  - `line.start` / `line.end` (each is a `Pt(x,y,z)`)
  - `line.segments` (each segment has `start`, `end`, `speed_label`)
- `scan` (scan only) – dict with:
  - `raw_start`, `raw_end` (Pt)
  - `buf_start`, `buf_end` (Pt)

it's recommended to pack z-hop logic in there if you require it.


## Analysis pipeline

`[rubidium analyse]` supports an optional, explicit pipeline ordering:

`analysis_pipeline: crop, brightness, clahe, blur, threshold, hsv_gate, morph, masked_gray, centroid`  
step names are:
- `crop`
- `brightness`
- `clahe`
- `blur`
- `threshold`
- `hsv_gate`
- `morph`
- `masked_gray`
- `centroid`

If `analysis_pipeline` is empty, Rubidium uses the default ordering above.

## Recommended workflow

1. Configure at least:
   - `[rubidium]`
   - `[rubidium print]`
   - `[rubidium scan]` with a working `video_source`
   - `[virtual_sdcard]`

2. Print the pattern:
   - `RUBIDIUM_PRINT PATTERN=my_pa_test`

3. Scan the pattern:
   - `RUBIDIUM_SCAN PATTERN=my_pa_test VIDEO_LATENCY_MS=...`

4. Analyse the most recent scan session:
   - `RUBIDIUM_ANALYSE`


## Example configured pattern and its results:

```
[rubidium pattern multistep]
origin_x:         100
origin_y:         100

layer_height:     0.25

line_length:      70
line_spacing:     2

pattern_segments:
    medium:         0.2,
    scv:            0.2,
    medium:         0.2,
    fast:           0.2,
    medium_slow:    0.2

param_start:      0.1
param_stop:       0.6
param_count:      12

tuning_command:   SET_PRESSURE_ADVANCE
tuning_parameter: OFFSET
```

<div style="display:flex; gap:12px;">
  <img src="media/analysis_5step.jpg" alt="Image 1" style="width:48%;">
  <img src="media/pattern_5step.jpg" alt="Image 2" style="width:48%;">
</div>


## Common gotchas

- If your `video_source` is a local V4L2 device (e.g. `/dev/v4l/by-id/...`), Rubidium requires:
  - `video_resolution`
  - `video_framerate`
- If you use speed labels in `pattern_segments`, you must define them in `speeds:` (or override via `SPEED_<LABEL>=...`).


## Status

Rubidium is an active work in progress.

---

Inspired by [Rubedo](<https://github.com/furrysalamander/rubedo>)  
