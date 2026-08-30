"""
YOLOv8 INT8 (LiteRT / TFLite) object-detection benchmark.

This is the CPU reference path. It runs the same `yolov8n_int8.tflite` on the
Raspberry Pi 5 (4x Cortex-A76) and on the Synaptics Astra SL2619 Arm CPU
(2x Cortex-A73) for an apples-to-apples comparison. Use `--threads 4` on the
Pi 5 and `--threads 2` on the SL2619 (one thread per core).

For the Astra Torq NPU, see `detect_torq.py` -- it shares everything in
`bench_common.py` with this script.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from bench_common import (
    add_common_args,
    decode_yolo_raw,
    letterbox,
    load_names,
    quantize_input,
    run_benchmark,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 INT8 (LiteRT / TFLite) object-detection benchmark "
        "(Raspberry Pi 5 / Astra SL2619 CPU)."
    )
    add_common_args(parser)
    parser.add_argument(
        "--model",
        default="../models/yolov8n_int8.tflite",
        help="Path to the YOLOv8 .tflite model (default: yolov8n_int8.tflite).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Model input size. Default: inferred from the .tflite input tensor.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Interpreter CPU threads (default: 4 = Pi 5 cores; use 2 on the SL2619).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# LiteRT / TFLite model wrapper
# ---------------------------------------------------------------------------

def _load_interpreter(model_path, num_threads):
    """Return a LiteRT/TFLite Interpreter, preferring the standalone runtimes."""
    last_err = None
    for module_name, attr in (
        ("ai_edge_litert.interpreter", "Interpreter"),
        ("tflite_runtime.interpreter", "Interpreter"),
        ("tensorflow.lite", "Interpreter"),
    ):
        try:
            mod = __import__(module_name, fromlist=[attr])
            Interpreter = getattr(mod, attr)
            return Interpreter(model_path=str(model_path), num_threads=num_threads)
        except ImportError as exc:
            last_err = exc
    raise RuntimeError(
        "No LiteRT/TFLite runtime found. Install one with:\n"
        "    pip install ai-edge-litert        # preferred\n"
        "    pip install tflite-runtime        # alternative\n"
        f"(last import error: {last_err})"
    )


class YoloV8TFLite:
    """
    Runs a YOLOv8 .tflite model exported by Ultralytics.

    Handles both export shapes:
      * a single [1, 84, 8400] (or [1, 8400, 84]) raw prediction tensor, and
      * the 4-tensor TFLite_Detection_PostProcess output (boxes/classes/scores/count).
    INT8 / UINT8 quantised inputs and outputs are (de)quantised automatically.
    """

    def __init__(self, model_path, num_threads=4, imgsz=None):
        self.interpreter = _load_interpreter(model_path, num_threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.outs = self.interpreter.get_output_details()

        shape = list(self.inp["shape"])
        # tflite is NHWC; guard against a stray NCHW export.
        self.nchw = shape[1] == 3 and shape[-1] != 3
        self.in_h = int(shape[2] if self.nchw else shape[1])
        self.in_w = int(shape[3] if self.nchw else shape[2])
        if imgsz:
            self.in_h = self.in_w = int(imgsz)
        self.imgsz = self.in_h

        self.in_dtype = self.inp["dtype"]
        self.in_scale, self.in_zero = self.inp["quantization"]
        self._raw_output = len(self.outs) < 4
        self.num_threads = num_threads
        self.device_infer_ms = None  # LiteRT has no separate device timer

    def describe(self):
        return (
            f"input {self.in_w}x{self.in_h}, "
            f"dtype {np.dtype(self.in_dtype).name}, "
            f"{'raw' if self._raw_output else 'NMS'} output, "
            f"{self.num_threads} threads"
        )

    # -- preprocessing ----------------------------------------------------

    def preprocess(self, frame):
        lb, r, dw, dh = letterbox(frame, self.in_w, self.in_h)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = quantize_input(x, self.in_dtype, self.in_scale, self.in_zero)
        x = np.expand_dims(x, 0)
        if self.nchw:
            x = np.transpose(x, (0, 3, 1, 2))
        self.interpreter.set_tensor(self.inp["index"], x)
        return r, dw, dh

    # -- inference ------------------------------------------------------

    def invoke(self):
        self.interpreter.invoke()

    def _dequant(self, detail):
        t = self.interpreter.get_tensor(detail["index"])
        scale, zero = detail["quantization"]
        if detail["dtype"] in (np.int8, np.uint8) and scale:
            return (t.astype(np.float32) - zero) * scale
        return t.astype(np.float32)

    # -- postprocessing ----------------------------------------------

    def postprocess(self, frame_shape, meta, conf_thres, iou_thres):
        r, dw, dh = meta
        if self._raw_output:
            pred = self._dequant(self.outs[0])[0]      # [84, 8400] or [8400, 84]
            return decode_yolo_raw(
                pred, self.imgsz, r, dw, dh, conf_thres, iou_thres,
                orig_hw=frame_shape,
            )

        # TFLite_Detection_PostProcess: locate outputs by shape.
        boxes_d = next(d for d in self.outs if len(d["shape"]) == 3
                       and d["shape"][-1] == 4)
        flat = [d for d in self.outs if d is not boxes_d and d["shape"][-1] != 4]
        count_d = min(flat, key=lambda d: int(np.prod(d["shape"])))
        rest = [d for d in flat if d is not count_d]
        scores_d = min(rest, key=lambda d: float(self._dequant(d).max()))
        classes_d = next(d for d in rest if d is not scores_d)

        n = int(self._dequant(count_d).reshape(-1)[0])
        ymin_xmin_ymax_xmax = self._dequant(boxes_d)[0][:n]
        confs = self._dequant(scores_d)[0][:n]
        class_ids = self._dequant(classes_d)[0][:n].astype(int)
        keep = confs >= conf_thres
        confs, class_ids = confs[keep], class_ids[keep]
        b = ymin_xmin_ymax_xmax[keep] * self.imgsz     # normalised -> pixels
        xyxy = np.stack([b[:, 1], b[:, 0], b[:, 3], b[:, 2]], axis=1)

        if not len(xyxy):
            return []

        h0, w0 = frame_shape[:2]
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - dw) / r).clip(0, w0)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - dh) / r).clip(0, h0)

        rects = [
            [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            for x1, y1, x2, y2 in xyxy
        ]
        idxs = cv2.dnn.NMSBoxes(rects, confs.tolist(), conf_thres, iou_thres)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).reshape(-1)
        return [(rects[i], int(class_ids[i]), float(confs[i])) for i in idxs]


def main():
    args = parse_args()

    if not Path(args.model).is_file():
        print(f"Error: model not found: {args.model}")
        raise SystemExit(1)

    names = load_names(args.labels)
    if names is None:
        print(f"Warning: labels file not found ({args.labels}); showing numeric class ids.")
    else:
        print(f"Loaded {len(names)} class names from {args.labels}")

    try:
        model = YoloV8TFLite(args.model, num_threads=args.threads, imgsz=args.imgsz)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
    args.imgsz = model.imgsz  # record the size actually used
    print(f"Loaded {args.model}  ({model.describe()})")

    run_benchmark(
        args,
        model,
        names,
        runtime="litert",
        window_title="YOLOv8 INT8 (LiteRT)",
        extra_metrics={"imgsz": model.imgsz, "threads": args.threads},
    )


if __name__ == "__main__":
    main()
