# Object detection benchmarks

Benchmark scripts that share their metrics, capture loop, YOLOv8 decode/NMS and
the optional INA219 power measurement (all in `bench_common.py`), differing only
in the inference back end:

| Script | Back end | Model | Platform |
| --- | --- | --- | --- |
| `detect_ultralytics.py` | Ultralytics YOLOv8 (PyTorch, CPU) | `.pt` (e.g. `yolov8n.pt`) | Pi 5 (reference path) |
| `detect_litert.py` | LiteRT / TFLite (CPU) | YOLOv8 `.tflite` (INT8 or float) | Pi 5 **and** Astra SL2619 CPU |
| `detect_torq.py` | Synaptics Torq runtime | YOLOv8 `.vmfb` (INT8) | Astra SL2619 Torq NPU (`--device torq`) or Torq CPU (`--device local-task`) |

`detect_litert.py` is deliberately the *same* script with the *same*
`yolov8n_int8.tflite` on both the Pi 5 and the Astra Arm CPU, so the CPU numbers
are directly comparable. Use one thread per core: `--threads 4` on the Pi 5
(4x Cortex-A76), `--threads 2` on the SL2619 (2x Cortex-A73). `detect_torq.py`
then measures the NPU with the identical loop and metrics.

Every script reports per-run inference time (avg / min / peak / median / p95),
FPS (inference-only, end-to-end, throughput), process memory (RSS), and — with
`--power` — average / peak / min board power, total energy, and energy per
inference. `detect_torq.py` additionally reports `device_inference_ms` (the
runtime-reported NPU time, excluding host overhead). Pass `--metrics-json PATH`
to also write everything to JSON.

# Setup

### Generating the YoloV8n INT8 model

This needs to be done on a Linux x86 or macOS with Python 3.12.

```sh
cd object_detection
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
```

```sh
python -m pip install --upgrade --force-reinstall \
  "torch==2.12.1" \
  "torchvision==0.27.1"
```

## Update target system & activate virtual environment

See top level [README.md](../README.md).

## Install dependencies

Shared:
```sh
# Raspberry Pi 5
pip install --no-cache-dir opencv-python numpy

# Astra SL2619 (Wayland/Weston, no X11) — use the headless build, or the
# system OpenCV via a --system-site-packages venv. Plain opencv-python pulls
# libxcb and fails at import with "libxcb.so.1: cannot open shared object file".
pip install --no-cache-dir opencv-python-headless numpy
```

For `detect_ultralytics.py`:
```sh
# cpu-only pytorch
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install --no-cache-dir ultralytics
pip install pandas-stubs
```

For `detect_litert.py`:
```sh
pip install --no-cache-dir ai-edge-litert      # or: tflite-runtime
```
On the Astra SL2619 CPU this is the same install — `tflite-runtime` has an
aarch64 wheel; `ai-edge-litert` also works.

For `detect_torq.py` (Astra SL2619 only):

Install torq runtime.
```sh
pip install https://github.com/synaptics-torq/torq-compiler/releases/download/v2.1.0/torq_runtime-2.1.0-cp312-cp312-manylinux_2_28_aarch64.whl
```

For `--power` on any script (INA219 over I2C):
```sh
pip install --no-cache-dir smbus2
```
On the Pi: enable I2C (`sudo raspi-config` -> Interface Options -> I2C) and wire
the INA219 in series with the Pi's 5 V USB-C input. The same works on the Astra
dev kit if an INA219 is wired in series with its 5 V input. Bus, address, shunt
value and sample rate are configured at the top of `../utils/pwr_meas.py`.

Confirm the wiring before a real run:
```sh
python3 ../utils/power_check.py            # chip check + plausibility + live V/I/P readout
python3 ../utils/power_check.py --sanity-only
```


## Connect USB camera

Connect the camera, then ensure it is recognized. 

```sh
lsusb
v4l2-ctl --list-devices
```

# Prepare a model

## Ultralytics
`yolov8n.pt` is downloaded automatically on first run, or pass `--model` with a
path to your own weights.

## LiteRT / TFLite
Export a YOLOv8 model to INT8 TFLite (needs Ultralytics installed):
```
yolo export model=yolov8n.pt format=tflite int8=True imgsz=320
```
This produces `yolov8n_full_integer_quant.tflite` (and metadata). Keep
`labels.json` (the `{"names": {...}}` sidecar) next to `detect_litert.py`; it is
loaded for class names by default.

## Torq NPU (`.vmfb`)
Download the pre-built Synaptics YOLOv8n Torq model + labels from Hugging Face:
```
python3 fetch_torq_model.py            # NPU model + labels.json
python3 fetch_torq_model.py --cpu      # also the local-task (CPU) vmfb
```
This drops the files under
`../models/Synaptics/yolov8-od-nano-320-int8-torq/`, the path `detect_torq.py`
expects by default. The model's quantisation parameters are baked into
`detect_torq.py`; override with `--in-scale/--in-zero/--out-scale/--out-zero`
or a `<model>.quant.json` sidecar if you use a different export.

# Run

## detect_ultralytics.py
```sh
python3 detect_ultralytics.py \
    --model yolov8n.pt \
    --source 0 \
    --width 320 --height 320 \
    --duration 60 \
    --no-display \
    --metrics-json ultra_run.json
```

## detect_litert.py  (Pi 5 CPU, and Astra SL2619 CPU — same command)
```sh
python3 detect_litert.py \
    --model ../models/yolov8n_int8.tflite \
    --source 0 \
    --threads 4 \
    --conf 0.25 --iou 0.45 \
    --duration 60 \
    --no-display \
    --metrics-json litert_run.json
```
On the Astra SL2619 use `--threads 2` (2 cores) and a `/dev/videoN` path for
`--source`.

## detect_torq.py  (Astra SL2619 Torq NPU)
```sh
python3 detect_torq.py \
    --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb \
    --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
    --source /dev/video0 \
    --conf 0.25 --iou 0.45 \
    --duration 60 \
    --no-display \
    --metrics-json torq_run.json
```
The script enables the NPU clock and pins the NPU to its max frequency for the
run, restoring min on exit (`--no-npu-setup` to skip; needs root for the
devfreq sysfs writes). Swap `--device local-task` (with the `--cpu` vmfb from
`fetch_torq_model.py`) for a Torq-runtime CPU baseline. For a live preview with
`--display` on the board, first:
```sh
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
```

## With power measurement (any script)
```sh
python3 detect_litert.py \
    --model yolov8n_full_integer_quant.tflite \
    --no-display \
    --duration 60 \
    --power \
    --idle-seconds 5 \
    --power-csv power.csv
```

Press `q` to quit the preview window; use `--no-display` for headless runs
(exit with Ctrl+C — the summary still prints).

# Options

## Common to all scripts (`bench_common.add_common_args`)

| Flag | Default | Description |
| --- | --- | --- |
| `--source` | `0` | Camera index or video file path. |
| `--width`, `--height` | `640x480` | Camera capture resolution. |
| `--max-frames N` | run until `q` | Stop after N measured frames. |
| `--duration SECS` | run until `q` | Stop after ~SECS seconds. |
| `--warmup N` | 5 | Initial frames excluded from the metrics. |
| `--no-display` | off | No preview window (headless benchmarking). |
| `--metrics-json PATH` | off | Also write metrics to JSON. |
| `--power` | off | Measure board power with an INA219 over I2C. |
| `--power-csv PATH` | off | Write every power sample to a CSV file. |
| `--idle-seconds N` | 0 | Measure idle power for N s first; also report idle-subtracted energy per inference. |
| `--conf` | 0.25 | Confidence threshold. |
| `--iou` | 0.45 | IoU threshold for non-max suppression. |
| `--labels PATH` | `labels.json` next to the script | Class names (`labels.json` or one name per line). |

`--model` is per-script (`yolov8n.pt` / `yolov8n_int8.tflite` /
`…/yolo_8n_2.0.0_npu.vmfb`). `detect_ultralytics.py` keeps its own `--conf`
(defaults to the Ultralytics model default) and does not take `--iou`/`--labels`.

## detect_litert.py only

| Flag | Default | Description |
| --- | --- | --- |
| `--imgsz` | inferred from the model | Model input size. |
| `--threads` | 4 | Interpreter CPU threads. Pi 5 = 4; SL2619 = 2. |

## detect_torq.py only

| Flag | Default | Description |
| --- | --- | --- |
| `--imgsz` | 320 | Model input size. |
| `--device` | `torq` | `torq` (NPU) or `local-task` (CPU vmfb). |
| `--tda` | `cpu` | Torq buffer allocator: `cpu` or `dmabuf`. `dmabuf` needs a working `/dev/dma_heap` XRAM heap; where that is missing it fails at invoke with `xram_write: accessing memory out of bounds`. |
| `--device-io` / `--no-device-io` | off | Keep model outputs as device arrays. |
| `--no-npu-setup` | off | Skip the NPU clock-enable + frequency pin. |
| `--in-scale`, `--in-zero`, `--out-scale`, `--out-zero` | model defaults | Override quantisation parameters. |
