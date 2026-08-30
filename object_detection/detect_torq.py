"""
YOLOv8 INT8 object-detection benchmark for the Synaptics Astra SL2619 Torq NPU.

Runs a Torq `.vmfb` YOLOv8n model through `torq.runtime.VMFBInferenceRunner`.
Shares argument parsing, the capture/measure loop, the YOLOv8 decode + NMS, the
metrics and the optional INA219 power measurement with `detect_litert.py` via
`bench_common.py`, so the NPU and CPU numbers are directly comparable.

    python3 detect_torq.py \
        --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb \
        --labels ../models/Synaptics/yolov8-od-nano-320-int8-torq/labels.json \
        --source /dev/video0 --no-display --duration 60 --metrics-json torq_run.json

`--device local-task` runs the same pipeline on the Torq CPU vmfb instead, for a
runtime-matched CPU baseline (the LiteRT path is the primary CPU comparison).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bench_common import (
    add_common_args,
    decode_yolo_raw,
    letterbox,
    load_names,
    quantize_input,
    run_benchmark,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from npu import configure_npu_userspace_frequency, enable_npu_clock  # noqa: E402

# Quantisation parameters for Synaptics/yolov8-od-nano-320-int8-torq. These match
# the constants baked into the shipped sl2610-examples object-detection demo.
# Override on the command line, or drop a "<model>.quant.json" sidecar next to
# the .vmfb with {"in_scale":..., "in_zero":..., "out_scale":..., "out_zero":...}.
DEFAULT_IN_SCALE = 0.003921568859368563   # 1/255
DEFAULT_IN_ZERO = -128
DEFAULT_OUT_SCALE = 0.004194467328488827
DEFAULT_OUT_ZERO = -128


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 INT8 object-detection benchmark for the Astra SL2619 Torq NPU."
    )
    add_common_args(parser)
    parser.add_argument(
        "--model",
        default="../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb",
        help="Path to the Torq YOLOv8 .vmfb model.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Model input size (default: 320).",
    )
    parser.add_argument(
        "--device",
        default="torq",
        choices=["torq", "local-task"],
        help="Torq device URI: 'torq' (NPU) or 'local-task' (CPU vmfb). Default: torq.",
    )
    parser.add_argument(
        "--tda",
        default="cpu",
        choices=["dmabuf", "cpu"],
        help="Allocator backing Torq device buffers (default: cpu). 'dmabuf' "
        "needs a working /dev/dma_heap XRAM heap; on runtimes/images where that "
        "is missing it fails at invoke with 'xram_write: accessing memory out "
        "of bounds'.",
    )
    parser.add_argument(
        "--device-io",
        dest="device_io",
        action="store_true",
        default=False,
        help="Keep model outputs as device arrays (default: off, host numpy).",
    )
    parser.add_argument(
        "--no-device-io",
        dest="device_io",
        action="store_false",
        help="Explicitly disable device I/O (the default).",
    )
    parser.add_argument(
        "--no-npu-setup",
        action="store_true",
        help="Skip enabling the NPU clock and pinning its frequency.",
    )
    parser.add_argument("--in-scale", type=float, default=None)
    parser.add_argument("--in-zero", type=int, default=None)
    parser.add_argument("--out-scale", type=float, default=None)
    parser.add_argument("--out-zero", type=int, default=None)
    return parser.parse_args()


def _resolve_quant(args):
    """Merge quant params: CLI > <model>.quant.json sidecar > built-in defaults."""
    q = {
        "in_scale": DEFAULT_IN_SCALE,
        "in_zero": DEFAULT_IN_ZERO,
        "out_scale": DEFAULT_OUT_SCALE,
        "out_zero": DEFAULT_OUT_ZERO,
    }
    sidecar = Path(args.model).with_suffix(".quant.json")
    if sidecar.is_file():
        q.update({k: v for k, v in json.loads(sidecar.read_text()).items() if k in q})
        print(f"Loaded quantisation parameters from {sidecar}")
    for cli_key, q_key in (
        ("in_scale", "in_scale"), ("in_zero", "in_zero"),
        ("out_scale", "out_scale"), ("out_zero", "out_zero"),
    ):
        val = getattr(args, cli_key)
        if val is not None:
            q[q_key] = val
    return q


# ---------------------------------------------------------------------------
# Torq VMFB model wrapper
# ---------------------------------------------------------------------------

class YoloV8Torq:
    """Runs a YOLOv8 Torq `.vmfb` model (raw [1, 84, N] prediction output)."""

    def __init__(self, model_path, *, device="torq", tda="dmabuf", device_io=True,
                 imgsz=320, quant=None):
        from torq.runtime import VMFBInferenceRunner

        quant = quant or {}
        self.in_scale = quant.get("in_scale", DEFAULT_IN_SCALE)
        self.in_zero = quant.get("in_zero", DEFAULT_IN_ZERO)
        self.out_scale = quant.get("out_scale", DEFAULT_OUT_SCALE)
        self.out_zero = quant.get("out_zero", DEFAULT_OUT_ZERO)

        self.imgsz = imgsz
        self.device = device
        self.tda = tda
        self.device_io = device_io
        self.device_infer_ms = None
        self._raw = None

        self.runner = VMFBInferenceRunner(
            model_path,
            device_uri=device,
            function="main",
            load_model_to_mem=True,
            device_outputs=device_io,
            runtime_flags=[f"--torq_device_allocator={tda}"],
        )

    def describe(self):
        return (
            f"device={self.device}, tda={self.tda}, device-io={self.device_io}, "
            f"input {self.imgsz}x{self.imgsz}"
        )

    # -- preprocessing --------------------------------------------------
    # Matches sl2610-examples object_detection_video.preprocess_frame_cv and
    # bench_common.letterbox used by detect_litert.py (grey 114 pad, RGB, /255).

    def preprocess(self, frame):
        import cv2

        lb, r, dw, dh = letterbox(frame, self.imgsz, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = quantize_input(rgb, np.int8, self.in_scale, self.in_zero)
        self._inp = np.expand_dims(x, 0)               # (1, imgsz, imgsz, 3)
        return r, dw, dh

    # -- inference ----------------------------------------------------

    def invoke(self):
        out = self.runner.infer([self._inp])
        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "to_host"):
            out = out.to_host()
        self._raw = np.asarray(out)
        self.device_infer_ms = self.runner.infer_time_ms

    # -- postprocessing --------------------------------------------

    def postprocess(self, frame_shape, meta, conf_thres, iou_thres):
        r, dw, dh = meta
        pred = (self._raw.astype(np.float32) - self.out_zero) * self.out_scale
        return decode_yolo_raw(
            pred, self.imgsz, r, dw, dh, conf_thres, iou_thres,
            orig_hw=frame_shape,
        )


def main():
    args = parse_args()

    if not Path(args.model).is_file():
        print(f"Error: model not found: {args.model}")
        print("Run  python fetch_torq_model.py  to download it.")
        raise SystemExit(1)

    names = load_names(args.labels)
    if names is None:
        print(f"Warning: labels file not found ({args.labels}); showing numeric class ids.")
    else:
        print(f"Loaded {len(names)} class names from {args.labels}")

    npu_active = args.device == "torq" and not args.no_npu_setup
    if npu_active:
        for ok, message in (enable_npu_clock(), configure_npu_userspace_frequency("max")):
            print(f"[NPU] {message}")

    try:
        try:
            model = YoloV8Torq(
                args.model,
                device=args.device,
                tda=args.tda,
                device_io=args.device_io,
                imgsz=args.imgsz,
                quant=_resolve_quant(args),
            )
        except ImportError as exc:
            print(f"Error: the Torq runtime is not installed ({exc}).\n"
                  "Install it with  pip install torq-runtime  or use the Astra SDK "
                  "Python environment.")
            raise SystemExit(1)

        print(f"Loaded {args.model}  ({model.describe()})")

        run_benchmark(
            args,
            model,
            names,
            runtime="torq",
            window_title="YOLOv8 INT8 (Torq NPU)",
            extra_metrics={
                "imgsz": model.imgsz,
                "device": args.device,
                "tda": args.tda,
                "device_io": args.device_io,
            },
        )
    finally:
        if npu_active:
            ok, message = configure_npu_userspace_frequency("min")
            print(f"[NPU] {message}")


if __name__ == "__main__":
    main()
