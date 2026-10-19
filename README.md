# Synaptics Coralboard SL vs. Raspberry Pi 5 Benchmarking


> [!WARNING]
> **Disclaimer:** These benchmark scripts are provided "as is" for informational purposes only. Results depend on the specific hardware, software versions, models, and test conditions used, and may contain errors or not be reproducible elsewhere. Verify independently before relying on them for any decision. No warranty is given, and the authors accept no liability for their use.

> [!NOTE]
> The benchmarking scripts include optional automatic power measurement using an external INA219 power measurement circuit. This feature has not been thoroughly tested and has not been used to generate results.

## Update target system (Raspberry Pi 5 Only)
```sh
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    ninja-build
```

## Install virtual environment
```sh
python3 -m venv .venv --system-site-packages 
```

## Activate & update virtual environment
```sh
source .venv/bin/activate
pip install --upgrade pip
```

## Jump to directories and view readme files

- [object_detection/README.md](object_detection/README.md)

- [llm/README.md](llm/README.md)
