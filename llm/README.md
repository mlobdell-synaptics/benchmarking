# Setup 

## Update system 

```
sudo apt update
sudo apt install -y \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    ninja-build
```

## Set up virtual environment

```
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

## Upgrade pip, etc
python3 -m pip install --upgrade pip setuptools wheel

## Set CMAKE args for building the llama-cpp wheel
CMAKE_ARGS="-DGGML_NATIVE=ON" \
CMAKE_BUILD_PARALLEL_LEVEL=4 \

## Install llama-cpp
```
python3 -m pip install --no-cache-dir llama-cpp-python
```

# Download models to test

## Post Training Quantization (PTQ)
```
cd models
wget https://huggingface.co/jaewoo49/gemm-3-270m-it-Q4_K_M-GGUF/resolve/main/gemma-3-270m-it-q4_k_m.gguf

wget https://huggingface.com/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-270m-it-Q8_0.gguf
```

## Quantization Arware Training (QAT)
```
wget https://huggingface.co/ggml-org/gemma-3-270m-it-qat-GGUF/resolve/main/gemma-3-270m-it-qat-Q4_0.gguf

wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

# Run

Basic benchmark (TTFT, generation speed, end-to-end speed per question, plus a
summary). Results are written to JSON.

```
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

```
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

