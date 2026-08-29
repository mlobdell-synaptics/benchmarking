"""
Download the Torq YOLOv8n object-detection assets from Hugging Face.

Pulls the NPU `.vmfb`, `labels.json` (and, with --cpu, the CPU `.vmfb`) from
`Synaptics/yolov8-od-nano-320-int8-torq` into
`../models/Synaptics/yolov8-od-nano-320-int8-torq/`, the layout `detect_torq.py`
expects by default.

    python fetch_torq_model.py            # NPU model + labels
    python fetch_torq_model.py --cpu      # also the local-task CPU vmfb
"""

import argparse
from pathlib import Path

REPO_ID = "Synaptics/yolov8-od-nano-320-int8-torq"
NPU_MODEL = "yolo_8n_2.0.0_npu.vmfb"
CPU_MODEL = "yolov8n_cpu.vmfb"
LABELS = "labels.json"

DEST = Path(__file__).resolve().parent.parent / "models" / "Synaptics" / \
    "yolov8-od-nano-320-int8-torq"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true",
                        help="Also download the Torq CPU (local-task) vmfb.")
    parser.add_argument("--dest", default=str(DEST),
                        help=f"Destination directory (default: {DEST}).")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        )

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    files = [NPU_MODEL, LABELS]
    if args.cpu:
        files.append(CPU_MODEL)

    for name in files:
        path = hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=str(dest))
        print(f"  {name} -> {path}")

    print(f"\nDone. Point --model at {dest / NPU_MODEL}")


if __name__ == "__main__":
    main()
