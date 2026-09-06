"""
Shared scaffolding for the object-detection benchmarks.

`detect_litert.py` (LiteRT/TFLite, CPU) and `detect_torq.py` (Synaptics Torq
NPU) differ only in how a frame is turned into a raw YOLOv8 prediction tensor.
Everything around that -- argument parsing, the capture/measure loop, the
YOLOv8 decode + NMS, the metrics, and the optional INA219 power measurement --
lives here so the two platforms stay byte-for-byte comparable.

A *detector* is any object with this duck-typed interface:

    detector.imgsz            -> int, the (square) model input size
    detector.describe()       -> str, one line printed once after load
    detector.preprocess(frame)      -> meta   (stores the input internally)
    detector.invoke()               -> None   (runs inference; may set
                                               `.device_infer_ms`)
    detector.postprocess(frame_shape, meta, conf, iou)
                              -> list[([x, y, w, h], cls_id, score)]

`meta` is opaque to the loop; it is whatever `preprocess` needs to hand back to
`postprocess` (here always the letterbox transform `(r, dw, dh)`).
"""

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
# with the TFLite export) sitting next to the benchmark script.
DEFAULT_LABELS = Path(__file__).resolve().parent / "labels.json"

# Headless runs print a "still alive" status line this often (seconds).
# Set to 0 to silence the heartbeat.
PROGRESS_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def add_common_args(parser):
    """
    Add the flags shared by every back end. `--model`, `--imgsz` and `--threads`
    are intentionally left to each script because their defaults differ.
    """
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
    parser.add_argument(
        "--quiesce-display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the Weston/Wayland compositor for the run, then restart "
        "it afterward. Reduces background CPU/power overhead. If --no-display "
        "was not also passed, the preview window degrades to headless (a "
        "harmless cv2.error is caught). Default: enabled "
        "(--no-quiesce-display to skip).",
    )
    parser.add_argument(
        "--quiesce-wifi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Soft-block the WiFi radio for the run, then unblock it "
        "afterward. Reduces background CPU/power overhead. Leaves usb0 "
        "(the adb/NCM link) untouched. Default: enabled (--no-quiesce-wifi "
        "to skip).",
    )
    return parser


# ---------------------------------------------------------------------------
# Pre / post processing shared by the LiteRT and Torq back ends
# ---------------------------------------------------------------------------

def letterbox(frame, in_w, in_h, pad=114):
    """
    Resize `frame` into an `in_w` x `in_h` canvas keeping aspect ratio, padding
    the remainder with the flat colour `pad`. Returns
    `(canvas, r, dw, dh)` where `r` is the resize ratio and `dw`/`dh` the
    left/top padding in canvas pixels.
    """
    h, w = frame.shape[:2]
    r = min(in_h / h, in_w / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((in_h, in_w, 3), pad, dtype=np.uint8)
    dw, dh = (in_w - nw) // 2, (in_h - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def quantize_input(x01, dtype, scale, zero):
    """
    Quantise a float image in [0, 1] to `dtype` using `scale`/`zero`. Returns the
    array unchanged (as float32) when the model input is not quantised.
    """
    if dtype in (np.int8, np.uint8) and scale:
        info = np.iinfo(dtype)
        return np.clip(
            np.round(x01 / scale + zero), info.min, info.max
        ).astype(dtype)
    return x01.astype(np.float32)


def decode_yolo_raw(pred, imgsz, r, dw, dh, conf_thres, iou_thres, orig_hw=None):
    """
    Decode a raw YOLOv8 prediction tensor into
    `[([x, y, w, h], cls_id, score), ...]` in original-frame pixels.

    Handles either layout ([84, N] or [N, 84]) and either box scale (normalised
    xywh in [0, 1] or already in model-input pixels). `r`, `dw`, `dh` are the
    letterbox transform returned by `letterbox()`. When `orig_hw` (the original
    `(h, w)`) is given, boxes are clipped to the frame.
    """
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        return []
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                                   # -> [N, 84]

    boxes = pred[:, :4].copy()
    cls_scores = pred[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(len(cls_scores)), class_ids]
    keep = confs >= conf_thres
    boxes, class_ids, confs = boxes[keep], class_ids[keep], confs[keep]
    if not len(boxes):
        return []

    if boxes.max() <= 1.5:                              # normalised xywh
        boxes *= imgsz

    # xywh (centre) -> xyxy in the letterboxed image
    xy = boxes[:, :2]
    wh = boxes[:, 2:4]
    xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

    # undo the letterbox: back to original-frame pixels
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / r
    if orig_hw is not None:
        h0, w0 = orig_hw[:2]
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


# ---------------------------------------------------------------------------
# Process memory
# ---------------------------------------------------------------------------

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
        return maxrss / 1024 if sys.platform == "linux" else maxrss / (1024 * 1024)
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _json_default(o):
    """Coerce numpy scalars/arrays so `json.dump` accepts a metrics dict."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _stats_block(samples_ms):
    ordered = sorted(samples_ms)
    return {
        "average": round(statistics.fmean(samples_ms), 3),
        "min": round(ordered[0], 3),
        "peak": round(ordered[-1], 3),
        "median": round(statistics.median(samples_ms), 3),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "stdev": round(statistics.pstdev(samples_ms), 3),
    }


def build_metrics(
    args,
    *,
    runtime,
    inference_times_ms,
    loop_times_ms,
    rss_samples_mb,
    elapsed_s,
    extra=None,
    preprocess_times_ms=None,
    postprocess_times_ms=None,
    device_infer_times_ms=None,
    power_result=None,
    idle_power_mw=None,
):
    frames = len(inference_times_ms)
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "model": args.model,
        "source": args.source,
        "capture_resolution": f"{args.width}x{args.height}",
        "frames_measured": frames,
        "warmup_frames": args.warmup,
        "elapsed_seconds": round(elapsed_s, 3),
        "quiesce_display": args.quiesce_display,
        "quiesce_wifi": args.quiesce_wifi,
    }
    if extra:
        metrics.update(extra)

    if frames:
        metrics["inference_ms"] = _stats_block(inference_times_ms)
        avg_loop_ms = statistics.fmean(loop_times_ms)
        metrics["fps"] = {
            "inference_only": round(1000 / metrics["inference_ms"]["average"], 2),
            "end_to_end": round(1000 / avg_loop_ms, 2) if avg_loop_ms > 0 else None,
            "throughput": round(frames / elapsed_s, 2) if elapsed_s > 0 else None,
        }
        if device_infer_times_ms:
            metrics["device_inference_ms"] = _stats_block(device_infer_times_ms)
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
    print(f"Model:                {metrics['model']}")

    cfg = []
    if metrics.get("imgsz") is not None:
        cfg.append(f"imgsz {metrics['imgsz']}")
    if metrics.get("threads") is not None:
        cfg.append(f"{metrics['threads']} threads")
    if cfg:
        print(f"Config:               {', '.join(cfg)}")
    if metrics.get("device") is not None:
        dev = f"device {metrics['device']}"
        if metrics.get("tda") is not None:
            dev += f", tda {metrics['tda']}"
        if metrics.get("device_io") is not None:
            dev += f", device-io {metrics['device_io']}"
        print(f"Device:               {dev}")

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

    dev_inf = metrics.get("device_inference_ms")
    if dev_inf:
        print(f"Device inference avg: {dev_inf['average']:.2f} ms "
              f"(runtime-reported, excludes host overhead)")

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


# ---------------------------------------------------------------------------
# Capture + power helpers
# ---------------------------------------------------------------------------

def open_capture(args):
    """Open `args.source` as a camera index or a media path and return the cap."""
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        raise SystemExit(1)
    return cap


def setup_power(args):
    """Return `(monitor, sampler)` for INA219 power measurement, or `(None, None)`."""
    if not args.power:
        return None, None
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
    from pwr_meas import make_power_sampler

    return make_power_sampler(csv_file=args.power_csv)


def measure_idle(sampler, args):
    """Measure idle board power for `args.idle_seconds`; return avg mW or None."""
    if sampler is None or args.idle_seconds <= 0:
        return None
    print(f"Measuring idle power for {args.idle_seconds:.1f} s (leave the board idle)...")
    sampler.start("idle")
    time.sleep(args.idle_seconds)
    idle_result = sampler.stop()
    if idle_result is None:
        return None
    idle_power_mw = idle_result.total_avg_power_mw
    print(f"Idle power: {idle_power_mw:.1f} mW")
    return idle_power_mw


# ---------------------------------------------------------------------------
# The measured loop
# ---------------------------------------------------------------------------

def run_benchmark(args, detector, names, *, runtime, window_title, extra_metrics=None):
    """
    Drive `detector` over the frames from `args.source`, collect the metrics and
    print (and optionally write) them. Returns the metrics dict.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
    from system_services import disable_display, disable_wifi, enable_display, enable_wifi

    if args.quiesce_display:
        for _ok, message in disable_display():
            print(f"[services] {message}")
    if args.quiesce_wifi:
        for _ok, message in disable_wifi():
            print(f"[services] {message}")

    try:
        return _run_benchmark_loop(
            args, detector, names, runtime=runtime, window_title=window_title,
            extra_metrics=extra_metrics,
        )
    finally:
        if args.quiesce_wifi:
            for _ok, message in enable_wifi():
                print(f"[services] {message}")
        if args.quiesce_display:
            for _ok, message in enable_display():
                print(f"[services] {message}")


def _run_benchmark_loop(args, detector, names, *, runtime, window_title, extra_metrics=None):
    monitor, sampler = setup_power(args)
    cap = open_capture(args)
    idle_power_mw = measure_idle(sampler, args)

    display = not args.no_display
    if display:
        print("Starting video stream... Press 'q' to exit.")
    else:
        print("Starting video stream (headless)... Press Ctrl+C to exit.")

    inference_times_ms = []      # wall-clock around invoke(), measured frames only
    device_infer_times_ms = []   # runtime-reported, when the detector exposes it
    preprocess_times_ms = []
    postprocess_times_ms = []
    loop_times_ms = []
    rss_samples_mb = []
    fps_smoothed = 0.0
    frame_index = 0
    start_time = None

    # Headless heartbeat so a long run visibly proves it is alive.
    heartbeat = (not display) and PROGRESS_INTERVAL_S > 0
    wall_start = time.perf_counter()
    last_beat = wall_start

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                if frame_index == 0:
                    print(f"No frames from source {args.source!r}. Wrong /dev/videoN "
                          "node, or the camera needs a format/resolution it does not "
                          "support. Try `v4l2-ctl --list-devices` / another node.")
                else:
                    print("Source ended (no more frames).")
                break

            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            meta = detector.preprocess(frame)
            t1 = time.perf_counter()
            detector.invoke()
            t2 = time.perf_counter()
            detections = detector.postprocess(
                frame.shape, meta, args.conf, args.iou
            )
            t3 = time.perf_counter()

            preprocess_ms = (t1 - t0) * 1000
            inference_ms = (t2 - t1) * 1000
            postprocess_ms = (t3 - t2) * 1000
            loop_ms = (time.perf_counter() - loop_start) * 1000
            device_infer_ms = getattr(detector, "device_infer_ms", None)

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
                if device_infer_ms is not None:
                    device_infer_times_ms.append(device_infer_ms)
                sample = rss_mb()
                if sample is not None:
                    rss_samples_mb.append(sample)

            instant_fps = 1000 / loop_ms if loop_ms > 0 else 0.0
            fps_smoothed = (
                instant_fps
                if fps_smoothed == 0
                else 0.9 * fps_smoothed + 0.1 * instant_fps
            )

            if frame_index == 0 and not display:
                print(f"First frame OK: preprocess {preprocess_ms:.1f} ms, "
                      f"inference {inference_ms:.1f} ms, postprocess {postprocess_ms:.1f} ms, "
                      f"{len(detections)} detection(s). Running...", flush=True)

            now = time.perf_counter()
            if heartbeat and (now - last_beat) >= PROGRESS_INTERVAL_S:
                last_beat = now
                phase = "warmup" if not measuring else "measuring"
                avg = f"{statistics.fmean(inference_times_ms):.1f}" if inference_times_ms else "--"
                dev = ""
                if device_infer_ms is not None:
                    dev = f" | npu {device_infer_ms:.1f} ms"
                print(
                    f"[{now - wall_start:6.1f}s] frame {frame_index + 1} ({phase}) | "
                    f"infer {inference_ms:.1f} ms (avg {avg}){dev} | "
                    f"{fps_smoothed:.1f} fps | {len(detections)} obj",
                    flush=True,
                )

            if display:
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
                try:
                    cv2.imshow(window_title, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as exc:
                    print(f"Preview unavailable (headless OpenCV build?): {exc}\n"
                          "Continuing without display.")
                    display = False

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
    if display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    power_result = None
    if sampler is not None:
        power_result = sampler.stop()
        sampler.close()
    if monitor is not None:
        monitor.close()

    elapsed_s = (time.perf_counter() - start_time) if start_time is not None else 0.0
    metrics = build_metrics(
        args,
        runtime=runtime,
        inference_times_ms=inference_times_ms,
        loop_times_ms=loop_times_ms,
        rss_samples_mb=rss_samples_mb,
        elapsed_s=elapsed_s,
        extra=extra_metrics,
        preprocess_times_ms=preprocess_times_ms,
        postprocess_times_ms=postprocess_times_ms,
        device_infer_times_ms=device_infer_times_ms,
        power_result=power_result,
        idle_power_mw=idle_power_mw,
    )
    print_metrics(metrics)

    if args.metrics_json:
        with open(args.metrics_json, "w") as fh:
            json.dump(metrics, fh, indent=2, default=_json_default)
        print(f"\nMetrics written to {args.metrics_json}")

    return metrics
