# Object detection benchmarks

Two benchmark scripts, same metrics and the same optional INA219 power
measurement, different inference back ends:

| Script | Back end | Model | Notes |
| --- | --- | --- | --- |
| `detect_ultralytics.py` | Ultralytics YOLOv8 (PyTorch, CPU) | `.pt` (e.g. `yolov8n.pt`) | The original reference path. |
| `detect_litert.py` | LiteRT / TFLite | YOLOv8 `.tflite` (INT8 or float) | For quantized-model testing on the Pi. |

Both report per-run inference time (avg / min / peak / median / p95), FPS
(inference-only, end-to-end, throughput), process memory (RSS), and — with
`--power` — average / peak / min board power, total energy, and energy per
inference. Pass `--metrics-json PATH` to also write everything to JSON.

# Setup

## Update system
```
sudo apt update && sudo apt upgrade -y
```

## Enter directory and set up virtual environment
```
cd object_detection
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
```

## Install dependencies

Shared:
```
pip install --no-cache-dir opencv-python numpy
```

For `detect_ultralytics.py`:
```
# cpu-only pytorch
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install --no-cache-dir ultralytics
pip install pandas-stubs
```

For `detect_litert.py`:
```
pip install --no-cache-dir ai-edge-litert      # or: tflite-runtime
```

For `--power` on either script (INA219 over I2C):
```
pip install --no-cache-dir smbus2
```
Enable I2C (`sudo raspi-config` -> Interface Options -> I2C) and wire the INA219
in series with the Pi's 5 V USB-C input. Bus, address, shunt value and sample
rate are configured at the top of `../utils/pwr_meas.py`.

## Connect USB camera

```
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

# Run

## detect_ultralytics.py
```
python3 detect_ultralytics.py \
    --model yolov8n.pt \
    --source 0 \
    --width 320 --height 320 \
    --duration 60 \
    --no-display \
    --metrics-json ultra_run.json
```

## detect_litert.py
```
python3 detect_litert.py \
    --model yolov8n_full_integer_quant.tflite \
    --source 0 \
    --threads 4 \
    --conf 0.25 --iou 0.45 \
    --duration 60 \
    --no-display \
    --metrics-json litert_run.json
```

## With power measurement (either script)
```
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

## Common to both scripts

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `yolov8n.pt` / `yolov8n_int8.tflite` | Model path. |
| `--source` | `0` | Camera index or video file path. |
| `--width`, `--height` | see below | Camera capture resolution. |
| `--max-frames N` | run until `q` | Stop after N measured frames. |
| `--duration SECS` | run until `q` | Stop after ~SECS seconds. |
| `--warmup N` | 5 | Initial frames excluded from the metrics. |
| `--no-display` | off | No preview window (headless benchmarking). |
| `--metrics-json PATH` | off | Also write metrics to JSON. |
| `--power` | off | Measure board power with an INA219 over I2C. |
| `--power-csv PATH` | off | Write every power sample to a CSV file. |
| `--idle-seconds N` | 0 | Measure idle power for N s first; also report idle-subtracted energy per inference. |

`--width`/`--height` default to `320x320` for `detect_ultralytics.py` and
`640x480` for `detect_litert.py`.

## detect_ultralytics.py only

| Flag | Default | Description |
| --- | --- | --- |
| `--conf` | Ultralytics default | Confidence threshold passed to the model. |

## detect_litert.py only

| Flag | Default | Description |
| --- | --- | --- |
| `--imgsz` | inferred from the model | Model input size. |
| `--threads` | 4 | Interpreter CPU threads. |
| `--conf` | 0.25 | Confidence threshold (applied in post-processing). |
| `--iou` | 0.45 | IoU threshold for non-max suppression. |
| `--labels PATH` | `labels.json` next to the script | Class names (`labels.json` or one name per line). |
