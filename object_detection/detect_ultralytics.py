import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 object-detection benchmark for the Raspberry Pi 5."
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to (or name of) the YOLO model weights (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or path to a video file (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Capture frame width (default: 320).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Capture frame height (default: 320).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold passed to the model (default: Ultralytics default).",
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
    power_result=None,
    idle_power_mw=None,
):
    frames = len(inference_times_ms)
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "source": args.source,
        "resolution": f"{args.width}x{args.height}",
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

    import cv2
    from ultralytics import YOLO

    monitor = sampler = None
    idle_power_mw = None
    if args.power:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
        from pwr_meas import make_power_sampler

        monitor, sampler = make_power_sampler(csv_file=args.power_csv)

    model = YOLO(args.model)

    # A bare integer means a camera index; anything else is treated as a file path.
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        raise SystemExit(1)

    predict_kwargs = {"verbose": False}
    if args.conf is not None:
        predict_kwargs["conf"] = args.conf

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

            results = model(frame, **predict_kwargs)

            inference_ms = results[0].speed["inference"]
            loop_ms = (time.perf_counter() - loop_start) * 1000

            measuring = frame_index >= args.warmup
            if measuring:
                if start_time is None:
                    start_time = time.perf_counter()
                    if sampler is not None:
                        sampler.start("detect")
                inference_times_ms.append(inference_ms)
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
                annotated_frame = results[0].plot()
                cv2.putText(
                    annotated_frame,
                    f"Inference: {inference_ms:.1f} ms | FPS: {fps_smoothed:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(
                    "YOLOv8 COCO Object Detection - Raspberry Pi 5", annotated_frame
                )
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
