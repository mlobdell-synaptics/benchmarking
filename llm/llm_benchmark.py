#!/usr/bin/env python3
"""
Benchmark a LLM GGUF model with llama-cpp-python.

Measures:
- TTFT: time from request submission until the first non-empty text token/chunk
- Generation speed: output tokens / time from first token until completion
- End-to-end speed: output tokens / total request time

Example:
    python3 llm_benchmark.py \
        --model <model.gguf> \
        --questions questions.txt \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # import lazily in main() so llm_benchmark_torq.py can reuse
    from llama_cpp import Llama  # this module's helpers without llama-cpp installed

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


def rss_mb() -> float | None:
    """Current resident set size of this process in MiB, or None if unavailable."""
    if _PROC is not None:
        return _PROC.memory_info().rss / (1024 * 1024)
    return None


def peak_rss_mb() -> float | None:
    """Peak RSS of this process (since start) in MiB, or None if unavailable.

    Captures the model-load spike even for questions sampled after it, since
    `ru_maxrss` is a running high-water mark for the whole process lifetime.
    """
    if resource is not None:
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS/BSD report bytes.
        return maxrss / 1024 if sys.platform == "linux" else maxrss / (1024 * 1024)
    return None


DEFAULT_QUESTIONS = [
    "What are the three primary colors of light?",
    "Explain the difference between RAM and storage in two sentences.",
    "A farmer has 17 sheep and all but 9 run away. How many remain?",
    "Write a Python function that returns True when an integer is even.",
    "Summarize why quantized models are useful on edge devices.",
]


@dataclass
class BenchmarkResult:
    question_number: int
    question: str
    response: str
    prompt_tokens: int
    output_tokens: int
    ttft_seconds: float
    generation_seconds: float
    total_seconds: float
    tokens_per_second: float
    end_to_end_tokens_per_second: float
    # Populated only by llm_benchmark_torq.py: the Torq runtime's own timing,
    # for cross-checking the wall-clock numbers above.
    runner_ttft_seconds: float | None = None
    runner_generation_seconds: float | None = None
    # Populated only when --power is active.
    avg_power_mw: float | None = None
    peak_power_mw: float | None = None
    min_power_mw: float | None = None
    energy_j: float | None = None
    energy_per_token_mj: float | None = None
    dynamic_energy_j: float | None = None
    dynamic_energy_per_token_mj: float | None = None
    power_sample_rate_hz: float | None = None
    # Process RSS sampled during this question's generation (best-effort; None
    # if psutil/resource aren't available on the target).
    avg_rss_mb: float | None = None
    peak_rss_mb: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GGUF inference using llama-cpp-python."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the GGUF file.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        help="Optional text file containing one question per non-empty line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSON output path. Default: "
            "<model_name>_benchmark_results.json (or _N suffix if that exists)."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="CPU threads used for generation. Default: up to 4.",
    )
    parser.add_argument(
        "--batch-threads",
        type=int,
        default=None,
        help="CPU threads used for prompt processing. Default: same as --threads.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=2048,
        help="Context size. Default: 2048.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum output tokens per question. Default: 128.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0 for repeatable tests.",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one unreported warm-up request. Default: enabled.",
    )
    parser.add_argument(
        "--verbose-model",
        action="store_true",
        help="Show llama.cpp model-loading and runtime logs.",
    )
    parser.add_argument(
        "--power",
        action="store_true",
        help="Measure board power per question with an INA219 on the I2C bus "
        "(bus/address/shunt/rate are configured at the top of utils/pwr_meas.py).",
    )
    parser.add_argument(
        "--power-csv",
        type=Path,
        default=None,
        help="Write every power sample to this CSV file.",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=0.0,
        help="Measure idle power for this long before the run, and also report "
        "dynamic (idle-subtracted) energy per question and per token.",
    )
    return parser.parse_args()


def unique_output_path(path: Path) -> Path:
    """Return `path`, or the first `<stem>_N<suffix>` that doesn't exist yet."""
    if not path.exists():
        return path

    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def load_questions(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_QUESTIONS.copy()

    if not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {path}")

    questions = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not questions:
        raise ValueError(f"No questions found in: {path}")
    return questions


def token_count(llm: Llama, text: str, *, add_bos: bool = False) -> int:
    """Count model tokens while remaining compatible with several API versions."""
    encoded = text.encode("utf-8")
    try:
        return len(llm.tokenize(encoded, add_bos=add_bos, special=True))
    except TypeError:
        return len(llm.tokenize(encoded, add_bos=add_bos))


def extract_stream_text(chunk: dict[str, Any]) -> str:
    """Extract generated text from a streamed chat-completion chunk."""
    choices = chunk.get("choices") or []
    if not choices:
        return ""

    choice = choices[0]
    delta = choice.get("delta") or {}

    # Standard streamed chat-completion format.
    content = delta.get("content")
    if isinstance(content, str):
        return content

    # Defensive fallback for API/version variations.
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def benchmark_question(
    llm: Llama,
    question: str,
    question_number: int,
    max_tokens: int,
    temperature: float,
    print_stream: bool = True,
    sampler: Any = None,
    idle_power_mw: float | None = None,
) -> BenchmarkResult:
    messages = [
        {
            "role": "system",
            "content": "Answer accurately and concisely.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # This is an estimate based on the message content. The model's embedded
    # chat template adds formatting/special tokens around these strings.
    prompt_text = "\n".join(str(message["content"]) for message in messages)
    prompt_tokens = token_count(llm, prompt_text, add_bos=True)

    first_text_time: float | None = None
    pieces: list[str] = []
    power = None
    rss_samples_mb: list[float] = []

    if sampler is not None:
        sampler.start(f"q{question_number}")
    start = time.perf_counter()
    try:
        stream: Iterable[dict[str, Any]] = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            stream=True,
        )

        for chunk in stream:
            text = extract_stream_text(chunk)
            if not text:
                continue

            now = time.perf_counter()
            if first_text_time is None:
                first_text_time = now

            pieces.append(text)
            if print_stream:
                print(text, end="", flush=True)

            sample = rss_mb()
            if sample is not None:
                rss_samples_mb.append(sample)

        end = time.perf_counter()
    finally:
        # Always end the session so a failed question can't leak into the next.
        if sampler is not None:
            power = sampler.stop()
    if print_stream:
        print()

    response = "".join(pieces)
    output_tokens = token_count(llm, response, add_bos=False) if response else 0

    if first_text_time is None:
        # No visible text was produced. Keep metrics valid and explicit.
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

    memory_fields: dict[str, float] = {}
    if rss_samples_mb:
        memory_fields["avg_rss_mb"] = statistics.fmean(rss_samples_mb)
        memory_fields["peak_rss_mb"] = max(rss_samples_mb)

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
        **memory_fields,
        **power_fields,
    )


def print_result(result: BenchmarkResult) -> None:
    print(
        f"\nTTFT:                {result.ttft_seconds:.3f} s\n"
        f"Output tokens:       {result.output_tokens}\n"
        f"Generation time:     {result.generation_seconds:.3f} s\n"
        f"Generation speed:    {result.tokens_per_second:.2f} tokens/s\n"
        f"Total request time:  {result.total_seconds:.3f} s\n"
        f"End-to-end speed:    {result.end_to_end_tokens_per_second:.2f} tokens/s"
    )
    if result.avg_rss_mb is not None:
        print(
            f"Avg RSS:             {result.avg_rss_mb:.1f} MiB\n"
            f"Peak RSS:            {result.peak_rss_mb:.1f} MiB"
        )
    if result.avg_power_mw is not None:
        print(
            f"Avg power:           {result.avg_power_mw:.1f} mW\n"
            f"Peak power:          {result.peak_power_mw:.1f} mW\n"
            f"Min power:           {result.min_power_mw:.1f} mW\n"
            f"Energy (request):    {result.energy_j:.3f} J "
            f"({result.power_sample_rate_hz:.0f} Hz sampling)\n"
            f"Energy / token:      {result.energy_per_token_mj:.2f} mJ"
        )
        if result.dynamic_energy_per_token_mj is not None:
            print(
                f"Dynamic energy:      {result.dynamic_energy_j:.3f} J "
                f"(idle-subtracted)\n"
                f"Dynamic E / token:   {result.dynamic_energy_per_token_mj:.2f} mJ"
            )


def build_summary(results: list[BenchmarkResult]) -> dict[str, float]:
    summary = {
        "question_count": len(results),
        "mean_ttft_seconds": statistics.fmean(r.ttft_seconds for r in results),
        "median_ttft_seconds": statistics.median(r.ttft_seconds for r in results),
        "mean_tokens_per_second": statistics.fmean(
            r.tokens_per_second for r in results
        ),
        "median_tokens_per_second": statistics.median(
            r.tokens_per_second for r in results
        ),
        "mean_total_seconds": statistics.fmean(r.total_seconds for r in results),
    }

    measured_mem = [r for r in results if r.avg_rss_mb is not None]
    if measured_mem:
        summary["mean_avg_rss_mb"] = statistics.fmean(
            r.avg_rss_mb for r in measured_mem
        )
        summary["max_peak_rss_mb"] = max(r.peak_rss_mb for r in measured_mem)

    powered = [r for r in results if r.avg_power_mw is not None]
    if powered:
        summary.update(
            {
                "mean_avg_power_mw": statistics.fmean(r.avg_power_mw for r in powered),
                "max_peak_power_mw": max(r.peak_power_mw for r in powered),
                "total_energy_j": sum(r.energy_j for r in powered),
                "mean_energy_per_token_mj": statistics.fmean(
                    r.energy_per_token_mj for r in powered
                ),
            }
        )
        if all(r.dynamic_energy_per_token_mj is not None for r in powered):
            summary["total_dynamic_energy_j"] = sum(
                r.dynamic_energy_j for r in powered
            )
            summary["mean_dynamic_energy_per_token_mj"] = statistics.fmean(
                r.dynamic_energy_per_token_mj for r in powered
            )
    return summary


def main() -> int:
    args = parse_args()

    from llama_cpp import Llama

    if not args.model.is_file():
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        return 2
    if args.threads < 1:
        print("Error: --threads must be at least 1.", file=sys.stderr)
        return 2
    if args.context < 256:
        print("Error: --context must be at least 256.", file=sys.stderr)
        return 2
    if args.max_tokens < 1:
        print("Error: --max-tokens must be at least 1.", file=sys.stderr)
        return 2

    try:
        questions = load_questions(args.questions)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    batch_threads = args.batch_threads or args.threads

    output_path = args.output or Path(f"{args.model.stem}_benchmark_results.json")
    output_path = unique_output_path(output_path)

    print(f"Loading model: {args.model}")
    print(f"Threads: {args.threads}; batch threads: {batch_threads}")
    print(f"Context: {args.context}; max output tokens: {args.max_tokens}")

    try:
        llm = Llama(
            model_path=str(args.model),
            n_ctx=args.context,
            n_threads=args.threads,
            n_threads_batch=batch_threads,
            n_batch=min(512, args.context),
            n_gpu_layers=0,       # Explicit CPU-only execution
            use_mmap=True,        # Helpful on memory-constrained systems
            use_mlock=False,
            verbose=args.verbose_model,
        )
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 1

    monitor = sampler = None
    idle_power_mw = None
    if args.power:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
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
            benchmark_question(
                llm=llm,
                question="Reply with only the word ready.",
                question_number=0,
                max_tokens=8,
                temperature=0.0,
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
            result = benchmark_question(
                llm=llm,
                question=question,
                question_number=index,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
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

    if sampler is not None:
        sampler.close()
    if monitor is not None:
        monitor.close()

    if not results:
        print("No benchmark requests completed successfully.", file=sys.stderr)
        return 1

    summary = build_summary(results)
    report = {
        "configuration": {
            "model": str(args.model.resolve()),
            "threads": args.threads,
            "batch_threads": batch_threads,
            "context": args.context,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "warmup": args.warmup,
            "cpu_only": True,
            "power_measured": sampler is not None,
            "idle_power_mw": idle_power_mw,
            # Whole-process high-water mark (since interpreter start), so it
            # also captures the model-load spike that per-question sampling
            # (after load) can't see. None if psutil/resource are unavailable.
            "peak_process_rss_mb": peak_rss_mb(),
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
    if "mean_avg_rss_mb" in summary:
        print(f"Mean avg RSS:            {summary['mean_avg_rss_mb']:.1f} MiB")
        print(f"Max peak RSS:            {summary['max_peak_rss_mb']:.1f} MiB")
    if report["configuration"]["peak_process_rss_mb"] is not None:
        print(
            f"Peak process RSS:        "
            f"{report['configuration']['peak_process_rss_mb']:.1f} MiB (incl. model load)"
        )
    if "mean_avg_power_mw" in summary:
        print(f"Mean power:              {summary['mean_avg_power_mw']:.1f} mW")
        print(f"Max peak power:          {summary['max_peak_power_mw']:.1f} mW")
        print(f"Total energy:            {summary['total_energy_j']:.2f} J")
        print(
            f"Mean energy per token:   {summary['mean_energy_per_token_mj']:.2f} mJ"
        )
        if "mean_dynamic_energy_per_token_mj" in summary:
            print(
                f"Mean dynamic E / token:  "
                f"{summary['mean_dynamic_energy_per_token_mj']:.2f} mJ (idle-subtracted)"
            )
    print(f"Results written to:      {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
