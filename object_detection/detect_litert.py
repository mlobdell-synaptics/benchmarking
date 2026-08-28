import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    import psutil

    _PROC = psutil.Process()
except Exception:  # psutil not installed on the target
    psutil = None
    _PROC = None

try:
    import resource
except ImportError:  # not available on Windows
    resource = None


# Class names come from labels.json (the Ultralytics metadata sidecar that ships
# with the TFLite export) sitting next to this script.
DEFAULT_LABELS = Path(__file__).resolve().parent / "labels.json"


def load_names(path):
    """
    Load class names from `path`. Returns a list indexed by class id, or None if
    the file is missing.

    Supports:
      * Ultralytics labels.json  -> {"names": {"0": "person", ...}}  (or a list)
      * a bare JSON list / dict
      * a plain text file with one name per line
    """
    path = Path(path)
    if not path.is_file():
        return None
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "names" in data:
            data = data["names"]
        if isinstance(data, dict):
            return [data[k] for k in sorted(data, key=lambda x: int(x))]
        return list(data)
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 INT8 (LiteRT / TFLite) object-detection benchmark for the "
        "Raspberry Pi 5."
    )
    parser.add_argument(
        "--model",
        default="../models/yolov8n_int8.tflite",
        help="Path to the YOLOv8 .tflite model (default: yolov8n_int8.tflite).",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or path to a video file (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera capture width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera capture height (default: 480).",
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
        help="Interpreter CPU threads (default: 4, the Pi 5 core count).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for non-max suppression (default: 0.45).",
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS),
        metavar="PATH",
        help=f"Class names: Ultralytics labels.json or one name per line "
        f"(default: {DEFAULT_LABELS.name} next to this script).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing this many frames (default: run until 'q').",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after roughly this many seconds (default: run until 'q').",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of initial frames excluded from the metrics (default: 5).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the preview window (useful for headless benchmarking).",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        metavar="PATH",
        help="Also write the collected metrics to this JSON file.",
    )
    parser.add_argument(
        "--power",
        action="store_true",
        help="Measure board power with an INA219 on the I2C bus (bus/address/"
        "shunt/rate are configured at the top of utils/pwr_meas.py).",
    )
    parser.add_argument(
        "--power-csv",
        default=None,
        metavar="PATH",
        help="Write every power sample to this CSV file.",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=0.0,
        help="Measure idle power for this long before the run, and also report "
        "dynamic (idle-subtracted) energy per inference.",
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
        self.in_h = shape[2] if self.nchw else shape[1]
        self.in_w = shape[3] if self.nchw else shape[2]
        if imgsz:
            self.in_h = self.in_w = imgsz
        self.imgsz = self.in_h

        self.in_dtype = self.inp["dtype"]
        self.in_scale, self.in_zero = self.inp["quantization"]
        self._raw_output = len(self.outs) < 4

    # -- preprocessing ----------------------------------------------------

    def letterbox(self, frame):
        h, w = frame.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        dw, dh = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw] = resized
        return canvas, r, dw, dh

    def preprocess(self, frame):
        lb, r, dw, dh = self.letterbox(frame)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        if self.in_dtype in (np.int8, np.uint8) and self.in_scale:
            info = np.iinfo(self.in_dtype)
            x = np.clip(
                np.round(x / self.in_scale + self.in_zero), info.min, info.max
            ).astype(self.in_dtype)
        else:
            x = x.astype(np.float32)
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

    def postprocess(self, frame_shape, r, dw, dh, conf_thres, iou_thres):
        h0, w0 = frame_shape[:2]
        if self._raw_output:
            pred = self._dequant(self.outs[0])[0]          # [84, 8400] or [8400, 84]
            if pred.shape[0] < pred.shape[1]:
                pred = pred.T                               # -> [8400, 84]
            boxes = pred[:, :4].copy()
            cls_scores = pred[:, 4:]
            class_ids = np.argmax(cls_scores, axis=1)
            confs = cls_scores[np.arange(len(cls_scores)), class_ids]
            keep = confs >= conf_thres
            boxes, class_ids, confs = boxes[keep], class_ids[keep], confs[keep]
            if boxes.size and boxes.max() <= 1.5:           # normalised xywh
                boxes *= self.imgsz
            # xywh (centre) -> xyxy in the letterboxed image
            xy = boxes[:, :2]
            wh = boxes[:, 2:4]
            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            xyxy = np.concatenate([x1y1, x2y2], axis=1)
        else:
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

        # undo the letterbox: to original-frame pixels
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / r
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0)

        rects = [
            [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            for x1, y1, x2, y2 in xyxy
        ]
        idxs = cv2.dnn.NMSBoxes(rects, confs.tolist(), conf_thres, iou_thres)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).reshape(-1)
        return [(rects[i], int(class_ids[i]), float(confs[i])) for i in idxs]


def rss_mb():
    """Current resident set size of this process in MiB, or None if unavailable."""
    if _PROC is not None:
        return _PROC.memory_info().rss / (1024 * 1024)
    return None


def peak_rss_mb():
    """Peak resident set size of this process in MiB, or None if unavailable."""
    if resource is not None:
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS/BSD report bytes.
        import sys

        return maxrss / 1024 if sys.platform == "linux" else maxrss / (1024 * 1024)
    return None


def build_metrics(
    args,
    inference_times_ms,
    loop_times_ms,
    rss_samples_mb,
    elapsed_s,
    preprocess_times_ms=None,
    postprocess_times_ms=None,
    power_result=None,
    idle_power_mw=None,
):
    frames = len(inference_times_ms)
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime": "litert",
        "model": args.model,
        "imgsz": args.imgsz,
        "threads": args.threads,
        "source": args.source,
        "capture_resolution": f"{args.width}x{args.height}",
        "frames_measured": frames,
        "warmup_frames": args.warmup,
        "elapsed_seconds": round(elapsed_s, 3),
    }

    if frames:
        ordered = sorted(inference_times_ms)
        metrics["inference_ms"] = {
            "average": round(statistics.fmean(inference_times_ms), 3),
            "min": round(ordered[0], 3),
            "peak": round(ordered[-1], 3),
            "median": round(statistics.median(inference_times_ms), 3),
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
            "stdev": round(statistics.pstdev(inference_times_ms), 3),
        }
        avg_loop_ms = statistics.fmean(loop_times_ms)
        metrics["fps"] = {
            "inference_only": round(1000 / metrics["inference_ms"]["average"], 2),
            "end_to_end": round(1000 / avg_loop_ms, 2) if avg_loop_ms > 0 else None,
            "throughput": round(frames / elapsed_s, 2) if elapsed_s > 0 else None,
        }
        if preprocess_times_ms and postprocess_times_ms:
            metrics["stage_ms"] = {
                "preprocess_avg": round(statistics.fmean(preprocess_times_ms), 3),
                "inference_avg": metrics["inference_ms"]["average"],
                "postprocess_avg": round(statistics.fmean(postprocess_times_ms), 3),
            }

    mem = {"peak_rss_mb": None if peak_rss_mb() is None else round(peak_rss_mb(), 1)}
    if rss_samples_mb:
        mem["average_rss_mb"] = round(statistics.fmean(rss_samples_mb), 1)
        mem["max_sampled_rss_mb"] = round(max(rss_samples_mb), 1)
    metrics["memory"] = mem

    if power_result is not None and frames:
        energy_total_mj = power_result.total_energy_mj
        power = {
            "avg_power_mw": round(power_result.total_avg_power_mw, 1),
            "peak_power_mw": round(power_result.total_peak_power_mw, 1),
            "min_power_mw": round(sum(power_result.min_power_mw.values()), 1),
            "sample_rate_hz": round(power_result.sample_rate_hz, 1),
            "samples": power_result.n_samples,
            "energy_total_j": round(energy_total_mj / 1000, 4),
            "energy_per_inference_mj": round(energy_total_mj / frames, 3),
        }
        if idle_power_mw is not None:
            net_mj = energy_total_mj - idle_power_mw * power_result.duration_s
            power["idle_power_mw"] = round(idle_power_mw, 1)
            power["dynamic_energy_per_inference_mj"] = round(
                max(0.0, net_mj) / frames, 3
            )
        metrics["power"] = power

    return metrics


def print_metrics(metrics):
    print("\n===== Benchmark metrics =====")
    print(f"Runtime:              {metrics['runtime']}")
    print(f"Model:                {metrics['model']}  (imgsz {metrics['imgsz']}, "
          f"{metrics['threads']} threads)")
    print(f"Frames measured:      {metrics['frames_measured']} "
          f"(after {metrics['warmup_frames']} warm-up)")
    print(f"Elapsed:              {metrics['elapsed_seconds']:.2f} s")

    inf = metrics.get("inference_ms")
    if inf:
        print(f"Inference avg:        {inf['average']:.2f} ms")
        print(f"Inference peak:       {inf['peak']:.2f} ms")
        print(f"Inference min:        {inf['min']:.2f} ms")
        print(f"Inference median:     {inf['median']:.2f} ms")
        print(f"Inference p95:        {inf['p95']:.2f} ms")

    stage = metrics.get("stage_ms")
    if stage:
        print(f"Preprocess avg:       {stage['preprocess_avg']:.2f} ms")
        print(f"Postprocess avg:      {stage['postprocess_avg']:.2f} ms")

    fps = metrics.get("fps")
    if fps:
        print(f"FPS (inference only): {fps['inference_only']:.2f}")
        if fps["end_to_end"] is not None:
            print(f"FPS (end-to-end):     {fps['end_to_end']:.2f}")
        if fps["throughput"] is not None:
            print(f"FPS (throughput):     {fps['throughput']:.2f}")

    mem = metrics.get("memory", {})
    if mem.get("average_rss_mb") is not None:
        print(f"Memory avg RSS:       {mem['average_rss_mb']:.1f} MiB")
        print(f"Memory max RSS:       {mem['max_sampled_rss_mb']:.1f} MiB")
    if mem.get("peak_rss_mb") is not None:
        print(f"Memory peak RSS:      {mem['peak_rss_mb']:.1f} MiB")

    pwr = metrics.get("power")
    if pwr:
        print(f"Power avg:            {pwr['avg_power_mw']:.1f} mW")
        print(f"Power peak:           {pwr['peak_power_mw']:.1f} mW")
        print(f"Power min:            {pwr['min_power_mw']:.1f} mW")
        if "idle_power_mw" in pwr:
            print(f"Power idle:           {pwr['idle_power_mw']:.1f} mW")
        print(f"Energy total:         {pwr['energy_total_j']:.4f} J "
              f"({pwr['samples']} samples @ {pwr['sample_rate_hz']:.0f} Hz)")
        print(f"Energy / inference:   {pwr['energy_per_inference_mj']:.3f} mJ")
        if "dynamic_energy_per_inference_mj" in pwr:
            print(f"  (idle-subtracted):  {pwr['dynamic_energy_per_inference_mj']:.3f} mJ")
    print("=============================")


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
    print(f"Loaded {args.model}  (input {model.in_w}x{model.in_h}, "
          f"dtype {np.dtype(model.in_dtype).name}, "
          f"{'raw' if model._raw_output else 'NMS'} output, {args.threads} threads)")

    monitor = sampler = None
    idle_power_mw = None
    if args.power:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
        from pwr_meas import make_power_sampler

        monitor, sampler = make_power_sampler(csv_file=args.power_csv)

    # A bare integer means a camera index; anything else is treated as a file path.
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        raise SystemExit(1)

    if sampler is not None and args.idle_seconds > 0:
        print(f"Measuring idle power for {args.idle_seconds:.1f} s (leave the board idle)...")
        sampler.start("idle")
        time.sleep(args.idle_seconds)
        idle_result = sampler.stop()
        if idle_result is not None:
            idle_power_mw = idle_result.total_avg_power_mw
            print(f"Idle power: {idle_power_mw:.1f} mW")

    if args.no_display:
        print("Starting video stream (headless)... Press Ctrl+C to exit.")
    else:
        print("Starting video stream... Press 'q' to exit.")

    inference_times_ms = []  # measured frames only (warm-up excluded)
    preprocess_times_ms = []
    postprocess_times_ms = []
    loop_times_ms = []
    rss_samples_mb = []
    fps_smoothed = 0.0
    frame_index = 0
    start_time = None

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            r, dw, dh = model.preprocess(frame)
            t1 = time.perf_counter()
            model.invoke()
            t2 = time.perf_counter()
            detections = model.postprocess(
                frame.shape, r, dw, dh, args.conf, args.iou
            )
            t3 = time.perf_counter()

            preprocess_ms = (t1 - t0) * 1000
            inference_ms = (t2 - t1) * 1000
            postprocess_ms = (t3 - t2) * 1000
            loop_ms = (time.perf_counter() - loop_start) * 1000

            measuring = frame_index >= args.warmup
            if measuring:
                if start_time is None:
                    start_time = time.perf_counter()
                    if sampler is not None:
                        sampler.start("detect")
                inference_times_ms.append(inference_ms)
                preprocess_times_ms.append(preprocess_ms)
                postprocess_times_ms.append(postprocess_ms)
                loop_times_ms.append(loop_ms)
                sample = rss_mb()
                if sample is not None:
                    rss_samples_mb.append(sample)

            instant_fps = 1000 / loop_ms if loop_ms > 0 else 0.0
            fps_smoothed = (
                instant_fps
                if fps_smoothed == 0
                else 0.9 * fps_smoothed + 0.1 * instant_fps
            )

            if not args.no_display:
                for (x, y, w, h), cls_id, score in detections:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    name = names[cls_id] if names and cls_id < len(names) else cls_id
                    label = f"{name} {score:.2f}"
                    cv2.putText(
                        frame, label, (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
                cv2.putText(
                    frame,
                    f"Inference: {inference_ms:.1f} ms | FPS: {fps_smoothed:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("YOLOv8 INT8 (LiteRT) - Raspberry Pi 5", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

            if args.max_frames is not None and len(inference_times_ms) >= args.max_frames:
                break
            if (
                args.duration is not None
                and start_time is not None
                and (time.perf_counter() - start_time) >= args.duration
            ):
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")

    cap.release()
    cv2.destroyAllWindows()

    power_result = None
    if sampler is not None:
        power_result = sampler.stop()
        sampler.close()
    if monitor is not None:
        monitor.close()

    elapsed_s = (time.perf_counter() - start_time) if start_time is not None else 0.0
    metrics = build_metrics(
        args,
        inference_times_ms,
        loop_times_ms,
        rss_samples_mb,
        elapsed_s,
        preprocess_times_ms=preprocess_times_ms,
        postprocess_times_ms=postprocess_times_ms,
        power_result=power_result,
        idle_power_mw=idle_power_mw,
    )
    print_metrics(metrics)

    if args.metrics_json:
        with open(args.metrics_json, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"\nMetrics written to {args.metrics_json}")


if __name__ == "__main__":
    main()
