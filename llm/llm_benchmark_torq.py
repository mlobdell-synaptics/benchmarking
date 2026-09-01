#!/usr/bin/env python3
"""
Benchmark a Torq (VMFB) LLM on the Synaptics Astra SL2619 NPU.

The Torq counterpart of llm_benchmark.py: same metrics (TTFT, generation tok/s,
end-to-end tok/s, and - with --power - energy per token from the INA219), same
JSON report shape, so NPU and CPU LLM runs are directly comparable.

Model files come from https://huggingface.co/Synaptics/gemma-3-270m-it-torq via
`python fetch_torq_llm.py`.

Example:
    python3 llm_benchmark_torq.py \
        --model ../models/Synaptics/gemma-3-270m-it-torq \
        --threads 2 --max-tokens 128 \
        --questions questions.txt --output gemma3_it_torq.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Shared with the GGUF benchmark - identical metric definitions and report shape.
from llm_benchmark import (
    BenchmarkResult,
    build_summary,
    load_questions,
    print_result,
    unique_output_path,
)
from fetch_torq_llm import MODEL_FILE_SETS

if TYPE_CHECKING:
    from torq_llm import Gemma3Static

_UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Torq VMFB LLM on the Astra SL2619 NPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("../models/Synaptics/gemma-3-270m-it-torq"),
        help="Model directory (from fetch_torq_llm.py) or a specific .vmfb file.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        help="Text file with one question per non-empty line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Default: <model_dir>_torq_benchmark_results.json.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU threads for the non-NPU parts of the graph. "
        "Default: all cores (SL2619 = 2).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Stop each answer after this many streamed tokens. Default: 128.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Prompt+generation cap. Default: auto-detected from the model.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Truncate/pad each prompt to this many tokens. Default: unset.",
    )
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. Default: 0 (greedy, repeatable).")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p. Default: 1.0.")
    parser.add_argument("--top-k", type=int, default=64, help="Top-k. Default: 64.")
    parser.add_argument(
        "--kv-cache-window",
        type=int,
        default=2,
        metavar="N",
        help="Sliding-window KV cache: keep the last N entries when full. Default: 2.",
    )
    parser.add_argument(
        "--no-kv-cache-window",
        action="store_true",
        help="Disable the sliding window (generation stops when the cache fills).",
    )
    parser.add_argument(
        "--instruct-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the Gemma-3 chat template + system prompt. Default: enabled "
        "(the -it repo).",
    )
    parser.add_argument(
        "--tda",
        choices=["cpu", "dmabuf"],
        default="cpu",
        help="Torq buffer allocator. Default: cpu. 'dmabuf' needs a working "
        "/dev/dma_heap XRAM heap or inference fails with 'xram_write: accessing "
        "memory out of bounds'.",
    )
    parser.add_argument(
        "--device-io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep KV-cache tensors as device arrays. Default: disabled.",
    )
    parser.add_argument(
        "--no-npu-setup",
        action="store_true",
        help="Skip enabling the NPU clock and pinning its frequency.",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one unreported warm-up question. Default: enabled.",
    )
    parser.add_argument(
        "--power",
        action="store_true",
        help="Measure board power per question with an INA219 (see "
        "utils/pwr_meas.py).",
    )
    parser.add_argument("--power-csv", type=Path, default=None,
                        help="Write every power sample to this CSV file.")
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=0.0,
        help="Measure idle power for this long first; also report dynamic "
        "(idle-subtracted) energy per token.",
    )
    return parser.parse_args()


def resolve_model_file(path: Path) -> Path:
    """Return the primary .vmfb: `path` itself if a file, else the first known
    model-file-set inside the directory."""
    if path.is_file():
        return path
    if path.is_dir():
        for file_set in MODEL_FILE_SETS:
            if all((path / name).is_file() for name in file_set):
                return path / file_set[0]
        raise FileNotFoundError(
            f"No known model file set in {path}. Run: python fetch_torq_llm.py"
        )
    raise FileNotFoundError(f"Model path not found: {path}")


def benchmark_question_torq(
    gemma3: "Gemma3Static",
    question: str,
    question_number: int,
    max_tokens: int,
    *,
    print_stream: bool = True,
    sampler: Any = None,
    idle_power_mw: float | None = None,
) -> BenchmarkResult:
    prompt_tokens = len(gemma3.tokenize(question, "user"))

    first_text_time: float | None = None
    pieces: list[str] = []
    power = None

    if sampler is not None:
        sampler.start(f"q{question_number}")
    start = time.perf_counter()
    stream = gemma3.run_stream(question)
    try:
        for index, chunk in enumerate(stream):
            if chunk:
                now = time.perf_counter()
                if first_text_time is None:
                    first_text_time = now
                pieces.append(chunk)
                if print_stream:
                    print(chunk, end="", flush=True)
            if index + 1 >= max_tokens:
                break
        end = time.perf_counter()
    finally:
        stream.close()  # run the runner's finally (sets generated_tokens / timing)
        if sampler is not None:
            power = sampler.stop()
    if print_stream:
        print()

    response = "".join(pieces)
    output_tokens = gemma3.generated_tokens or max(0, len(pieces) - 1)

    if first_text_time is None:
        first_text_time = end

    ttft = first_text_time - start
    generation_seconds = max(0.0, end - first_text_time)
    total_seconds = end - start
    tokens_per_second = (
        output_tokens / generation_seconds if generation_seconds > 0 else 0.0
    )
    end_to_end_tokens_per_second = (
        output_tokens / total_seconds if total_seconds > 0 else 0.0
    )

    runner_ttft_seconds = gemma3.time_to_first_token / 1000.0
    runner_generation_seconds = max(
        0.0, (gemma3.last_infer_time - gemma3.time_to_first_token) / 1000.0
    )

    power_fields: dict[str, float] = {}
    if power is not None:
        energy_mj = power.total_energy_mj
        power_fields = {
            "avg_power_mw": power.total_avg_power_mw,
            "peak_power_mw": power.total_peak_power_mw,
            "min_power_mw": sum(power.min_power_mw.values()),
            "energy_j": energy_mj / 1000,
            "energy_per_token_mj": energy_mj / output_tokens if output_tokens else 0.0,
            "power_sample_rate_hz": power.sample_rate_hz,
        }
        if idle_power_mw is not None:
            dyn_mj = max(0.0, energy_mj - idle_power_mw * power.duration_s)
            power_fields["dynamic_energy_j"] = dyn_mj / 1000
            power_fields["dynamic_energy_per_token_mj"] = (
                dyn_mj / output_tokens if output_tokens else 0.0
            )

    return BenchmarkResult(
        question_number=question_number,
        question=question,
        response=response,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_seconds=ttft,
        generation_seconds=generation_seconds,
        total_seconds=total_seconds,
        tokens_per_second=tokens_per_second,
        end_to_end_tokens_per_second=end_to_end_tokens_per_second,
        runner_ttft_seconds=runner_ttft_seconds,
        runner_generation_seconds=runner_generation_seconds,
        **power_fields,
    )


def main() -> int:
    args = parse_args()

    try:
        model_file = resolve_model_file(args.model)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    model_dir = model_file.parent

    if args.threads < 1:
        print("Error: --threads must be at least 1.", file=sys.stderr)
        return 2
    if args.max_tokens < 1:
        print("Error: --max-tokens must be at least 1.", file=sys.stderr)
        return 2

    try:
        questions = load_questions(args.questions)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    output_path = args.output or Path(f"{model_dir.name}_torq_benchmark_results.json")
    output_path = unique_output_path(output_path)

    sys.path.insert(0, str(_UTILS_DIR))
    from npu import configure_npu_userspace_frequency, enable_npu_clock

    npu_active = not args.no_npu_setup
    if npu_active:
        for _ok, message in (
            enable_npu_clock(),
            configure_npu_userspace_frequency("max"),
        ):
            print(f"[NPU] {message}")

    try:
        try:
            from torq_llm import Gemma3Static
        except ImportError as exc:
            print(
                f"Error: the Torq runtime is not installed ({exc}).\n"
                "Install it with the torq_runtime wheel (see "
                "../object_detection/README.md) or use the Astra SDK Python env.",
                file=sys.stderr,
            )
            return 1

        print(f"Loading model: {model_file}")
        print(f"Threads: {args.threads}; tda: {args.tda}; device-io: {args.device_io}")
        print(f"Max output tokens: {args.max_tokens}; instruct: {args.instruct_model}")

        try:
            gemma3 = Gemma3Static(
                model_file,
                args.max_seq_len,
                max_prompt_tokens=args.max_input_tokens,
                n_threads=args.threads,
                instruct_model=args.instruct_model,
                cache_keep_n=None if args.no_kv_cache_window else args.kv_cache_window,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                #runtime_flags=[f"--torq_device_allocator={args.tda}"],
                device_io=args.device_io,
            )
        except Exception as exc:
            print(f"Failed to load model: {exc}", file=sys.stderr)
            return 1
        print(f"Max sequence length: {gemma3.max_seq_len}")

        monitor = sampler = None
        idle_power_mw = None
        if args.power:
            sys.path.insert(0, str(_UTILS_DIR))
            from pwr_meas import make_power_sampler

            monitor, sampler = make_power_sampler(
                csv_file=str(args.power_csv) if args.power_csv else None
            )
            if sampler is not None and args.idle_seconds > 0:
                print(f"\nMeasuring idle power for {args.idle_seconds:.1f} s...")
                sampler.start("idle")
                time.sleep(args.idle_seconds)
                idle_result = sampler.stop()
                if idle_result is not None:
                    idle_power_mw = idle_result.total_avg_power_mw
                    print(f"Idle power: {idle_power_mw:.1f} mW")

        if args.warmup:
            print("\nRunning warm-up request...")
            try:
                benchmark_question_torq(
                    gemma3,
                    "Reply with only the word ready.",
                    question_number=0,
                    max_tokens=8,
                    print_stream=False,
                )
            except Exception as exc:
                print(f"Warm-up failed: {exc}", file=sys.stderr)
                return 1

        results: list[BenchmarkResult] = []
        for index, question in enumerate(questions, start=1):
            print(f"\n{'=' * 72}")
            print(f"Question {index}/{len(questions)}: {question}")
            print("-" * 72)
            print("Response:")
            try:
                result = benchmark_question_torq(
                    gemma3,
                    question,
                    question_number=index,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    idle_power_mw=idle_power_mw,
                )
            except KeyboardInterrupt:
                print("\nBenchmark interrupted.", file=sys.stderr)
                return 130
            except Exception as exc:
                print(f"\nQuestion failed: {exc}", file=sys.stderr)
                continue
            results.append(result)
            print_result(result)
            if result.runner_ttft_seconds is not None:
                print(
                    f"Runtime TTFT:        {result.runner_ttft_seconds:.3f} s "
                    f"(NPU-reported)\n"
                    f"Runtime generation:  {result.runner_generation_seconds:.3f} s "
                    f"(NPU-reported)"
                )

        if sampler is not None:
            sampler.close()
        if monitor is not None:
            monitor.close()
    finally:
        if npu_active:
            _ok, message = configure_npu_userspace_frequency("min")
            print(f"[NPU] {message}")

    if not results:
        print("No benchmark requests completed successfully.", file=sys.stderr)
        return 1

    summary = build_summary(results)
    report = {
        "configuration": {
            "model": str(model_file.resolve()),
            "runtime": "torq",
            "tda": args.tda,
            "device_io": args.device_io,
            "threads": args.threads,
            "max_seq_len": gemma3.max_seq_len,
            "max_tokens": args.max_tokens,
            "kv_cache_window": None if args.no_kv_cache_window else args.kv_cache_window,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "instruct_model": args.instruct_model,
            "warmup": args.warmup,
            "power_measured": sampler is not None,
            "idle_power_mw": idle_power_mw,
        },
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"Questions completed:     {summary['question_count']}")
    print(f"Mean TTFT:               {summary['mean_ttft_seconds']:.3f} s")
    print(f"Median TTFT:             {summary['median_ttft_seconds']:.3f} s")
    print(f"Mean generation speed:   {summary['mean_tokens_per_second']:.2f} tokens/s")
    print(
        f"Median generation speed: "
        f"{summary['median_tokens_per_second']:.2f} tokens/s"
    )
    if "mean_avg_power_mw" in summary:
        print(f"Mean power:              {summary['mean_avg_power_mw']:.1f} mW")
        print(f"Max peak power:          {summary['max_peak_power_mw']:.1f} mW")
        print(f"Total energy:            {summary['total_energy_j']:.2f} J")
        print(f"Mean energy per token:   {summary['mean_energy_per_token_mj']:.2f} mJ")
        if "mean_dynamic_energy_per_token_mj" in summary:
            print(
                f"Mean dynamic E / token:  "
                f"{summary['mean_dynamic_energy_per_token_mj']:.2f} mJ (idle-subtracted)"
            )
    print(f"Results written to:      {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
