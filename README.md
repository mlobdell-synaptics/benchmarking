# Coralboard vs. Raspberry Pi 5 Benchmarking


> [!WARNING]
> **Disclaimer:** This is for information only. It could be wrong.


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
