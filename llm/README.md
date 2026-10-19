# Setup 

## Update target system & activate virtual environment

See top level [README.md](../README.md).

## Install dependencies

Pick the requirements file for your target:

| Target | File |
| --- | --- |
| Raspberry Pi 5 | `requirements-pi5.txt` |
| Astra SL2610 | `requirements-astra-sl2610.txt` |
| macOS | `requirements-macos.txt` |

```sh
pip install --upgrade setuptools wheel
CMAKE_ARGS="-DGGML_NATIVE=ON" CMAKE_BUILD_PARALLEL_LEVEL=4 \
    pip install -r requirements-<target>.txt
```

Raspberry Pi 5 and Astra SL2610 install a prebuilt CPU wheel from `../wheels/`
(no compile needed, `CMAKE_ARGS` is a no-op there). macOS has no prebuilt
wheel, so `llama-cpp-python` builds from source — `CMAKE_ARGS` is what
actually takes effect there; see the comments in `requirements-macos.txt` for
the Metal-support flag (unused by `llm_benchmark.py` as written, which
hardcodes CPU-only execution).

Set HuggingFace access token for downloads

```sh
export HF_TOKEN=...
```

# CPU - Model Download and testing

## GGUF Q4_K_M (PTQ - Post Training Quantization)

Download the model
```sh
wget -P ./models https://huggingface.co/gguf-org/gemma-3-270m-it-gguf/resolve/main/gemma-3-270m-it-q4_k_m.gguf
```

Test the model
```sh
python3 llm_benchmark.py \
    --model ../models/gemma-3-270m-it-q4_k_m.gguf \
    --threads 2 --context 256 --max-tokens 128 \
    --questions questions.txt \
    --output raspi_cpux2_gemma3_270m_it_q4_k_m.json

python3 llm_benchmark.py \
    --model ../models/gemma-3-270m-it-q4_k_m.gguf \
    --threads 4 --context 256 --max-tokens 128 \
    --questions questions.txt \
    --output raspi_cpux4_gemma3_270m_it_Q4_k_m.json
```

## GGUF Q8_0

Download the model
```sh
wget -P ./models https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf
```

Test the model
```sh
python3 llm_benchmark.py \
    --model ../models/gemma-3-270m-it-Q8_0.gguf \
    --threads 2 --context 256 --max-tokens 128 \
    --questions questions.txt \
    --output raspi_cpux2_gemma3_270m_it_Q8.json

python3 llm_benchmark.py \
    --model ../models/gemma-3-270m-it-Q8_0.gguf \
    --threads 4 --context 256 --max-tokens 128 \
    --questions questions.txt \
    --output raspi_cpux4_gemma3_270m_it_Q8.json
```

## GGUF BF16 (full precision)

`llm_benchmark.py` takes any GGUF path, so a BF16 comparison run needs no code
changes — just a BF16 weight file.

Gemma 3 270M-it: bartowski publishes a real BF16 file (the `ggml-org` repo
above only has Q8_0).
```sh
wget -P ./models https://huggingface.co/bartowski/google_gemma-3-270m-it-GGUF/resolve/main/google_gemma-3-270m-it-bf16.gguf
```

Run them the same way as any other GGUF (see Run, below):
```sh
cd llm
python3 llm_benchmark.py \
    --model ../models/google_gemma-3-270m-it-bf16.gguf \
    --threads 2 --context 256 --max-tokens 128 \
    --questions questions.txt \
    --output raspi_cpux2_gemma3_270m_it_bf16.json

python3 llm_benchmark.py \
    --model ../models/google_gemma-3-270m-it-bf16.gguf \
    --threads 4 --context 256 --max-tokens 128 \
    --questions questions.txt \
    --output raspi_cpux4_gemma3_270m_it_bf16.json
```

## FP32 (full precision)

FP32 is 2x BF16 and provides little accuracy improvement, however it runs more efficiently on Raspberry Pi. 

```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
pip install -U "huggingface_hub[cli]"

hf download google/gemma-3-270m-it --local-dir gemma-3-270m-it-hf

python3 llama.cpp/convert_hf_to_gguf.py gemma-3-270m-it-hf \
    --outtype f32 \
    --outfile gemma-3-270m-it-f32.gguf
```

```sh
cd llm
## F32
python3 llm_benchmark.py \
    --model ../models/gemma-3-270m-it-f32.gguf \
    --threads 2 --context 256 --max-tokens 128 \
    --questions questions.txt \
    --output raspi_cpux2_gemma3_270m_it_f32.json

python3 llm_benchmark.py \
    --model ../models/gemma-3-270m-it-f32.gguf \
    --threads 4 --context 256 --max-tokens 128 \
    --questions questions.txt \
    --output raspi_cpux4_gemma3_270m_it_f32.json
```

## Run

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
    --model ../<model path>\
    --threads 2 \
    --context 256 \
    --max-tokens 128 \
    --questions questions.txt \
    --output results.json
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
| `--quiesce-display` / `--no-quiesce-display` | on | Stop the display (Weston on Astra, `lightdm` on Raspberry Pi OS Desktop) for the run, restart on exit (no-op if neither is present, e.g. headless Pi). |
| `--quiesce-wifi` / `--no-quiesce-wifi` | on | Soft-block WiFi (`rfkill`) for the run, unblock on exit. Leaves Ethernet and `usb0`/adb untouched. |
| `--quiesce-network` | off | Stop the full network stack (NetworkManager/dhcpcd/networkd/wpa_supplicant/dnsmasq — Ethernet *and* WiFi) for the run, restart on exit. Only use over a connection that doesn't depend on IP networking (e.g. a serial console) — it will drop an SSH session before it can restore anything. |

On the Astra boards, adb/serial shells already run as root, so `--quiesce-*`
just works. On a non-root login (e.g. Raspberry Pi OS Desktop's default
user), stopping/starting services needs privilege: `systemctl` calls go
through `sudo -n`, which succeeds silently if the account has (passwordless)
sudo rights — the Raspberry Pi Imager default user does — and otherwise fails
fast with a clear message rather than hanging on an unanswerable polkit
prompt. See the top of `utils/system_services.py` for the sudoers.d rule to
add if your account isn't already passwordless-sudo. `rfkill` (used by
`--quiesce-wifi`) is tried unprivileged first and typically just works
(seat-permitted via udev on Raspberry Pi OS Desktop).

# Torq NPU (Astra SL2619)

`llm_benchmark_torq.py` is the Torq-NPU counterpart of `llm_benchmark.py` — same
metrics, same JSON report shape (plus `runtime: "torq"` and NPU-reported
`runner_*` timings), so NPU and CPU LLM runs are directly comparable. It runs a
pre-compiled VMFB model; the Torq runner code is vendored under `torq_llm/`
(from `torq-examples`).

## Install

```bash
pip install --upgrade pip

pip install torq-runtime

pip install -r requirements-astra-sl2610.txt
```

## Download and Run


### BF16
```sh
python3 fetch_torq_llm.py                 # Synaptics/gemma-3-270m-it-torq
```
Default lands in `../models/Synaptics/gemma-3-270m-it-torq/` (VMFB + tokenizer/config/embeddings). 

```sh
python3 llm_benchmark_torq.py \
    --model ../models/Synaptics/gemma-3-270m-it-torq \
    --threads 2 \
    --max-tokens 128 \
    --questions questions.txt \
    --output torq_gemma3_it_bf16.json
```

### Q8
To fetch int8 gemma3 270m for Torq 

```sh
python3 fetch_torq_llm.py --model gemma3-it-q8 #Synaptics/gemma-3-270m-it-int8-torq
```

```sh
python3 llm_benchmark_torq.py \
    --model ../models/Synaptics/gemma-3-270m-it-int8-torq \
    --threads 2 \
    --max-tokens 128 \
    --questions questions.txt \
    --output torq_gemma3_it_q8.json
```

Alternative - Dynamic Quantization

```sh
python3 fetch_torq_llm.py --model gemma3-it-q8-dynamic #Synaptics/gemma-3-270m-it-int8-dynamic-quantize-torq
```
temporary workaround: swap out token_embeddings.npy with one from the other int8 repo. 

```sh
python3 llm_benchmark_torq.py \
    --model ../models/Synaptics/gemma-3-270m-it-int8-dynamic-quantize-torq \
    --threads 2 \
    --max-tokens 128 \
    --questions questions.txt \
    --output torq_gemma3_it_q8_dynamic.json
```


### Q4

```sh
python3 fetch_torq_llm.py --model gemma3-it-q4 #Synaptics/gemma-3-270-GPTQ-int4-it-torq
```

```sh
python3 llm_benchmark_torq.py \
    --model ../models/Synaptics/gemma-3-270-GPTQ-int4-it-torq \
    --threads 2 \
    --max-tokens 128 \
    --questions questions.txt \
    --output torq_gemma3_it_q4.json
```


## Run


```sh
python3 llm_benchmark_torq.py \
    --model ../<model path> \
    --threads 2 \
    --max-tokens 128 \
    --questions questions.txt \
    --output resilts.json
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
| `--tda {cpu,dmabuf}` | `dmabuf` | Torq buffer allocator (zero-copy NPU access). Needs a working `/dev/dma_heap` XRAM heap; fall back to `cpu` if inference fails with `xram_write: accessing memory out of bounds`. |
| `--device-io` / `--no-device-io` | off | Keep KV-cache tensors as device arrays. |
| `--no-npu-setup` | off | Skip the NPU clock-enable + frequency pin. |
| `--quiesce-display` / `--no-quiesce-display` | on | Stop Weston for the run, restart on exit. |
| `--quiesce-wifi` / `--no-quiesce-wifi` | on | Soft-block WiFi (`rfkill`) for the run, unblock on exit. Leaves `usb0`/adb untouched. |
| `--temperature` / `--top-p` / `--top-k` | 0.0 / 1.0 / 64 | Sampling. |

## Raspberry Pi 5 config

To enable the serial connections on the 40-pin header (pins 14/15). 

Open the boot config file: 
```sh
sudo vim /boot/firmware/config.txt
```

Add these lines to the bottom of the file:
```sh
enable_uart=1
dtoverlay=miniuart-bt
```

Run the config utility and enable messages to the serial port:
```sh
sudo raspi-config
```

Go to Interface Options -> Serial Port.

Select Yes to get the login shell over serial.

Select Yes to enable serial port hardware.

Click Finish to exit and reboot.

