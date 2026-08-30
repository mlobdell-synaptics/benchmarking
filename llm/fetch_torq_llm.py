"""
Download a Torq (VMFB) LLM from Hugging Face for llm_benchmark_torq.py.

Fetches the model file set + the tokenizer/config/embeddings sidecars into
`../models/Synaptics/<repo-basename>/`, the layout `llm_benchmark_torq.py`
expects.

    python fetch_torq_llm.py                # Synaptics/gemma-3-270m-it-torq
    python fetch_torq_llm.py --model gemma3 # Synaptics/gemma-3-270m-torq (base)

Mirrors torq-examples/gemma3/setup_demo.py without its utils/ dependency.
"""

import argparse
from pathlib import Path

# Add a Qwen entry here when its Torq VMFB is published.
REPO_MAP = {
    "gemma3-it": "Synaptics/gemma-3-270m-it-torq",
    "gemma3": "Synaptics/gemma-3-270m-torq",
}

# First set whose every file exists on the repo is downloaded. The primary
# model file is the first of the pair; the second (if any) is the LM head.
MODEL_FILE_SETS = (
    ("transformer.vmfb", "lm_head.vmfb.trim"),
    ("transformer.vmfb", "lm_head.vmfb"),
    ("model.vmfb.trim",),
    ("model.vmfb",),
)
REQUIRED_FILES = ("token_embeddings.npy", "config.json", "tokenizer.json")
OPTIONAL_FILES = ("token_id_lut.npy",)

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models" / "Synaptics"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="gemma3-it",
        help="Model key from REPO_MAP or a raw HF repo id "
        f"(keys: {', '.join(REPO_MAP)}). Default: gemma3-it.",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination directory (default: ../models/Synaptics/<repo-basename>/).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is required: pip install huggingface_hub")

    repo_id = REPO_MAP.get(args.model, args.model)
    dest = Path(args.dest) if args.dest else MODELS_ROOT / repo_id.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = set(api.list_repo_files(repo_id=repo_id))

    model_set = next(
        (s for s in MODEL_FILE_SETS if all(f in repo_files for f in s)), None
    )
    if model_set is None:
        raise SystemExit(
            f"No known model file set found in {repo_id}. Tried: "
            + " | ".join("+".join(s) for s in MODEL_FILE_SETS)
        )

    wanted = list(model_set) + list(REQUIRED_FILES)
    wanted += [f for f in OPTIONAL_FILES if f in repo_files]

    missing = [f for f in wanted if f not in repo_files]
    if missing:
        raise SystemExit(f"{repo_id} is missing expected files: {', '.join(missing)}")

    print(f"Downloading {repo_id} -> {dest}")
    for name in wanted:
        path = hf_hub_download(repo_id=repo_id, filename=name, local_dir=str(dest))
        print(f"  {name} -> {path}")

    print(f"\nDone. Run:\n  python3 llm_benchmark_torq.py --model {dest}")


if __name__ == "__main__":
    main()
