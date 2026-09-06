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

## BF16 (full precision)

`llm_benchmark.py` takes any GGUF path, so a BF16 comparison run needs no code
changes — just a BF16 weight file.

Gemma 3 270M-it: bartowski publishes a real BF16 file (the `ggml-org` repo
above only has Q8_0).
```sh
wget https://huggingface.co/bartowski/google_gemma-3-270m-it-GGUF/resolve/main/google_gemma-3-270m-it-bf16.gguf
```

Qwen2.5-0.5B-Instruct: no repo (official or bartowski) publishes a BF16 file,
only `f16` (fp16, not bf16). Convert it yourself from the original HF
checkpoint with llama.cpp's converter (separate from `llama-cpp-python`):
```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
pip install -U "huggingface_hub[cli]"

hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir qwen2.5-0.5b-instruct-hf

python3 llama.cpp/convert_hf_to_gguf.py qwen2.5-0.5b-instruct-hf \
    --outtype bf16 \
    --outfile qwen2.5-0.5b-instruct-bf16.gguf
```

Note: BF16 files are ~2x the Q8_0 size, and on CPUs without native BF16 SIMD
kernels `ggml` typically upconverts to F32 for compute — so BF16 speed can
land close to (or behind) F16/Q8_0 rather than being "full precision but
fast." That's expected, not a bug in the run.

Run them the same way as any other GGUF (see Run, below):
```sh
cd llm
python3 llm_benchmark.py \
    --model ../models/google_gemma-3-270m-it-bf16.gguf \
    --threads 2 --context 1024 --max-tokens 128 \
    --questions questions.txt \
    --output llama_cpp/gemma3_270m_it_bf16.json

python3 llm_benchmark.py \
    --model ../models/qwen2.5-0.5b-instruct-bf16.gguf \
    --threads 2 --context 1024 --max-tokens 128 \
    --questions questions.txt \
    --output llama_cpp/qwen2.5_0.5b_instruct_bf16.json
```

## FP32 (full precision)

No repo publishes an FP32 GGUF (it's ~2x the BF16 size for no accuracy
benefit), so convert it yourself from the original HF checkpoint the same way
as the Qwen BF16 case above, but with `--outtype f32`:
```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
pip install -U "huggingface_hub[cli]"

hf download google/gemma-3-270m-it --local-dir gemma-3-270m-it-hf

python3 llama.cpp/convert_hf_to_gguf.py gemma-3-270m-it-hf \
    --outtype f32 \
    --outfile gemma-3-270m-it-f32.gguf
```

Note: FP32 is ~4x the Q8_0 file size and ~2x BF16/F16, so expect the slowest
load and highest RSS in the comparison table — that's the point of including
it. Unlike BF16 (which `ggml` often upconverts to F32 anyway on CPUs without
native BF16 kernels), this is the true "no quantization" baseline.

Run it the same way as any other GGUF (see Run, below):
```sh
cd llm
python3 llm_benchmark.py \
    --model ../models/gemma-3-270m-it-f32.gguf \
    --threads 2 --context 1024 --max-tokens 128 \
    --questions questions.txt \
    --output llama_cpp/gemma3_270m_it_f32.json
```

# Run

Basic benchmark (TTFT, generation speed, end-to-end speed per question, plus a
summary). Results are written to JSON. Process RSS (average and peak, sampled
during each question's generation) is also tracked automatically via `psutil`/
`resource` — no flag needed; the fields just come back `null` if `psutil`
isn't installed. `configuration.peak_process_rss_mb` additionally captures the
whole run's high-water mark, including the model-load spike that per-question
sampling (which only runs after load) can't see.

```sh
cd llm
python3 llm_benchmark.py \
    --model ../models/gemma-3-270m-it-[ver].gguf \
    --threads 2 \
    --context 1024 \
    --max-tokens 128 \
    --questions questions.txt \
    --output results.json
```

With INA219 power measurement (average / peak / min power, energy per request,
and mJ per output token, per question). Bus, address, shunt and sample rate are
configured at the top of `../utils/pwr_meas.py`. Verify the circuit first with
`python3 ../utils/power_check.py`.

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
| `--quiesce-display` / `--no-quiesce-display` | on | Stop Weston for the run, restart on exit (no-op if not present, e.g. headless Pi). |
| `--quiesce-wifi` / `--no-quiesce-wifi` | on | Soft-block WiFi (`rfkill`) for the run, unblock on exit. Leaves `usb0`/adb untouched. |

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

The Weston compositor and WiFi radio are stopped/soft-blocked for the run by
default (restoring both on exit), to remove their background CPU/power
overhead from the measurement — use `--no-quiesce-display` / `--no-quiesce-wifi`
to keep either running. WiFi is disabled via `rfkill block wifi`, not by
stopping NetworkManager/networkd — that also manages `usb0` (the
adb/NCM-gadget link), so it's left alone; run over `adb shell`, not SSH, if
WiFi quiescing is enabled. If a run is interrupted before its own cleanup
runs, restore both manually with `utils/system_services.py`:
```sh
python3 ../utils/system_services.py enable all      # or: display / wifi
```

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
| `--quiesce-display` / `--no-quiesce-display` | on | Stop Weston for the run, restart on exit. |
| `--quiesce-wifi` / `--no-quiesce-wifi` | on | Soft-block WiFi (`rfkill`) for the run, unblock on exit. Leaves `usb0`/adb untouched. |
| `--temperature` / `--top-p` / `--top-k` | 0.0 / 1.0 / 64 | Sampling. |

