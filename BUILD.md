# Building libtransformers


## Requirements

- CMake 3.21+
- C++20 compiler (GCC 11+, Clang 14+, MSVC 19.30+)
- CUDA Toolkit 11.8+ with a Volta or newer GPU (sm_70+)
- Internet access on first build (FetchContent pulls nlohmann/json and cpp-httplib)


For Framework support:
 - embeddings

Model Support TODO: 
 - deekseek
 - OSS
 - bunch of other ones 


## Build

```bash
cmake -B build
cmake --build build

```

To build the developer tools and tests:

```bash

apt-get update && apt-get install -y \
    pkg-config \
    linux-libc-dev \
    build-essential \
    make


# 1. Clone vcpkg into a permanent home (e.g. /opt/vcpkg or ~/vcpkg)
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg

# 2. Run the bootstrap script — builds the vcpkg binary
~/vcpkg/bootstrap-vcpkg.sh

# 3. Set VCPKG_ROOT so the cmake command you have works as-is
export VCPKG_ROOT=~/vcpkg
export PATH="$VCPKG_ROOT:$PATH"   # optional, lets you run vcpkg from anywhere


echo 'export VCPKG_ROOT=~/vcpkg' >> ~/.bashrc
echo 'export PATH="$VCPKG_ROOT:$PATH"' >> ~/.bashrc
source ~/.bashrc


cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DTRANSFORMERS_BUILD_TOOLS=ON
cmake --build build

```

## CMake options

| Option | Default | Description |
| --- | --- | --- |
| `TRANSFORMERS_BUILD_TOOLS` | `OFF` | Build developer tools and tests |

## Tools

| Binary | Description |
| --- | --- |
| `parser_safetensors` | Exercises every parser code path against a model file |
| `model-cli` | Resolves and downloads models from remote hubs (e.g., Hugging Face) |
| `run_llama` | End-to-end LLaMA inference, single prompt and chat modes |

---

## Tests

### Fetching Models

libtransformers includes `model-cli` to handle downloading and verifying weights. By default, models are cached globally in `~/.cache/models/`.

```bash
# ~17 MB  — tiny BERT, F32, fast sanity check
./build/model-cli fetch hf://gaunernst/bert-tiny-uncased

# ~90 MB  — MiniLM, F32, wider tensor variety
./build/model-cli fetch hf://sentence-transformers/all-MiniLM-L6-v2

# ~250 MB — OPT-125M, larger embedding table, good total_bytes stress
./build/model-cli fetch hf://facebook/opt-125m

```

Run the SafeTensors parser against the downloaded files:

```bash
./build/parser_safetensors ~/.cache/models/gaunernst/bert-tiny-uncased/model.safetensors
./build/parser_safetensors ~/.cache/models/sentence-transformers/all-MiniLM-L6-v2/model.safetensors
./build/parser_safetensors ~/.cache/models/facebook/opt-125m/model.safetensors

```

### LLaMA inference

Any LLaMA-architecture model in safetensors format works.

```bash
# ~2.5 GB  — LLaMA-3.2-1B, BF16, extremely fast for local iteration
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
    --prompt "What is the company Google ?" \
    --max-tokens 200 \
    --temperature 0.8

./build/run_llama ~/.cache/models/Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --prompt "What is the company Google ?" \
    --max-tokens 200 \
    --temperature 0.8

./build/run_llama ~/.cache/models/unsloth/gemma-2b-it \
    --prompt "What is the company Google ?" \
    --max-tokens 200 \
    --temperature 0.8

```

Run interactive chat:

```bash
./build/run_llama ~/.cache/models/unsloth/Llama-3.2-1B --chat --temperature 0.8

```