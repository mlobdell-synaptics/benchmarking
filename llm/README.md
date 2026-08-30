# Setup 

## Update target system & activate virtual environment

See top level [README.md](../README.md).

## Install dependencies
```sh
pip install --upgrade setuptools wheel
```

## Set CMAKE args for building the llama-cpp wheel
CMAKE_ARGS="-DGGML_NATIVE=ON" \
CMAKE_BUILD_PARALLEL_LEVEL=4 \

## Install llama-cpp
```sh
python3 -m pip install --no-cache-dir llama-cpp-python
```

# Download models to test

## Post Training Quantization (PTQ)
```sh
cd models
wget https://huggingface.co/jaewoo49/gemm-3-270m-it-Q4_K_M-GGUF/resolve/main/gemma-3-270m-it-q4_k_m.gguf

wget https://huggingface.com/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-270m-it-Q8_0.gguf
```

## Quantization Arware Training (QAT)
```sh
wget https://huggingface.co/ggml-org/gemma-3-270m-it-qat-GGUF/resolve/main/gemma-3-270m-it-qat-Q4_0.gguf

wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

# Run

Basic benchmark (TTFT, generation speed, end-to-end speed per question, plus a
summary). Results are written to JSON.

```sh
cd llm
python3 llm_benchmark.py \
    --model gemma-3-270m-it-[ver].gguf \
    --threads 4 \
    --context 1024 \
    --max-tokens 128 \
    --questions questions.txt \
    --output results.json
```

With INA219 power measurement (average / peak / min power, energy per request,
and mJ per output token, per question). Bus, address, shunt and sample rate are
configured at the top of `../utils/pwr_meas.py`.

```sh
python3 llm_benchmark.py \
    --model gemma-3-270m-it-[ver].gguf \
    --questions questions.txt \
    --power \
    --idle-seconds 5 \
    --power-csv power.csv
```

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | *(required)* | Path to the GGUF file. |
| `--questions` | built-in set | Text file, one prompt per non-empty line (`#` comments allowed). |
| `--output` | `<model>_benchmark_results.json` | JSON results path (`_N` suffix if it exists). |
| `--threads` | up to 4 | CPU threads for generation. |
| `--batch-threads` | same as `--threads` | CPU threads for prompt processing. |
| `--context` | 2048 | Context size (min 256). |
| `--max-tokens` | 128 | Max output tokens per question. |
| `--temperature` | 0.0 | Sampling temperature (0 = repeatable). |
| `--no-warmup` | warmup on | Skip the unreported warm-up request. |
| `--verbose-model` | off | Show llama.cpp load/runtime logs. |
| `--power` | off | Measure board power per question with an INA219 over I2C. |
| `--power-csv PATH` | off | Write every power sample to a CSV file. |
| `--idle-seconds N` | 0 | Measure idle power for N s first; also report idle-subtracted energy. |

# Torq NPU (Astra SL2619)

`llm_benchmark_torq.py` is the Torq-NPU counterpart of `llm_benchmark.py` — same
metrics, same JSON report shape (plus `runtime: "torq"` and NPU-reported
`runner_*` timings), so NPU and CPU LLM runs are directly comparable. It runs a
pre-compiled VMFB model; the Torq runner code is vendored under `torq_llm/`
(from `torq-examples`).

## Install

```sh
pip install -r requirements-torq.txt
# plus the Torq runtime wheel — see ../object_detection/README.md
```

## Download the model

```sh
python3 fetch_torq_llm.py                 # Synaptics/gemma-3-270m-it-torq
```
Lands in `../models/Synaptics/gemma-3-270m-it-torq/` (VMFB + tokenizer/config/
embeddings). A Qwen VMFB can be added later via `REPO_MAP` in that script
(`python3 fetch_torq_llm.py --model qwen`).

## Run

```sh
python3 llm_benchmark_torq.py \
    --model ../models/Synaptics/gemma-3-270m-it-torq \
    --threads 2 \
    --max-tokens 128 \
    --questions questions.txt \
    --output gemma3_it_torq.json
```
(`--threads` defaults to the core count anyway — 2 on the SL2619.)

The script enables the NPU clock and pins the NPU to max frequency for the run
(restoring min on exit; `--no-npu-setup` to skip — needs root for the devfreq
writes). `--power` / `--idle-seconds` / `--power-csv` behave exactly as in
`llm_benchmark.py`.

## Torq options (beyond the shared ones)

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `../models/Synaptics/gemma-3-270m-it-torq` | Model directory (from `fetch_torq_llm.py`) or a specific `.vmfb`. |
| `--max-tokens` | 128 | Stop each answer after this many streamed tokens (the runner has no native cap). |
| `--max-seq-len` | auto | Prompt+generation cap; auto-detected from the model. |
| `--max-input-tokens` | unset | Truncate/pad each prompt to this many tokens. |
| `--kv-cache-window N` / `--no-kv-cache-window` | 2 | Sliding-window KV cache size. |
| `--instruct-model` / `--no-instruct-model` | on | Gemma-3 chat template + system prompt. |
| `--tda {cpu,dmabuf}` | `cpu` | Torq buffer allocator. `dmabuf` needs a working `/dev/dma_heap` XRAM heap or inference fails with `xram_write: accessing memory out of bounds`. |
| `--device-io` / `--no-device-io` | off | Keep KV-cache tensors as device arrays. |
| `--no-npu-setup` | off | Skip the NPU clock-enable + frequency pin. |
| `--temperature` / `--top-p` / `--top-k` | 0.0 / 1.0 / 64 | Sampling. |

