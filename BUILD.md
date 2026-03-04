# Building tensor_inference

## Requirements

* CMake 3.21+
* C++20 compiler (GCC 11+, Clang 14+, MSVC 19.30+)
* CUDA Toolkit 11.8+ with a Volta or newer GPU (sm_70+)
* Internet access on first build (vcpkg pulls dependencies)

## Dependencies (Local vcpkg)

This project manages dependencies locally using a vendored `vcpkg` instance. You do not need a system-wide vcpkg installation.

1. **Install system prerequisites:**
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    pkg-config \
    curl \
    zip \
    unzip \
    tar \
    linux-libc-dev

```


2. **Clone and bootstrap vcpkg locally:**
From the root of the project:
```bash
git clone --depth 1 https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh

```


3. **Install libraries:**
```bash
./vcpkg/vcpkg install curl nlohmann-json openssl

```



## Build

Configure the project using the local toolchain file and build:

```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DTENSOR_BUILD_TOOLS=ON

cmake --build build --parallel

```

## CMake options

| Option | Default | Description |
| --- | --- | --- |
| `TENSOR_BUILD_TOOLS` | `OFF` | Build developer tools (`model-cli`, `run_llama`, etc.) |

## Tools

| Binary | Description |
| --- | --- |
| `parser_safetensors` | Exercises every parser code path against a model file |
| `model-cli` | Resolves and downloads models from remote hubs (e.g., Hugging Face) |
| `run_llama` | End-to-end LLaMA inference, single prompt and chat modes |

---

## Usage Examples

### Fetching Models

`tensor_inference` includes `model-cli` to handle downloading and verifying weights. By default, models are cached globally in `~/.cache/models/`.

```bash
# ~17 MB  — tiny BERT, F32, fast sanity check
./build/model-cli fetch hf://gaunernst/bert-tiny-uncased

# ~90 MB  — MiniLM, F32, wider tensor variety
./build/model-cli fetch hf://sentence-transformers/all-MiniLM-L6-v2

# ~250 MB — OPT-125M, larger embedding table
./build/model-cli fetch hf://facebook/opt-125m

```

### LLaMA Inference

Any LLaMA-architecture model in `.safetensors` format works.

```bash
# ~2.5 GB — LLaMA-3.2-1B, BF16, extremely fast for local iteration
# Requires a HuggingFace account and accepting the model license.
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
./build/model-cli fetch hf://unsloth/Llama-3.2-1B-Instruct
./build/model-cli fetch hf://Qwen/Qwen2.5-Coder-1.5B-Instruct
./build/model-cli fetch hf://unsloth/gemma-2b-it
./build/model-cli fetch hf://mistralai/Ministral-3B-instruct

```

Run a single prompt using the cached path:

```bash
./build/run_llama ~/.cache/models/unsloth/Llama-3.2-1B-Instruct \
    --prompt "What is the company Google?" \
    --max-tokens 200 \
    --temperature 0.8

./build/run_qwen ~/.cache/models/Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --prompt "Write a rust tcp server" \
    --max-tokens 200 \
    --temperature 0.8

```

Run interactive chat:

```bash
./build/run_llama ~/.cache/models/unsloth/Llama-3.2-1B-Instruct --chat --temperature 0.8

```