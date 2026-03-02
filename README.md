# tensor-inference

> Part of the **Tensor Framework** by Netangular

A native C++ inference engine for transformer-architecture models.
Load any model, run on any hardware, hot-swap knowledge adapters at runtime.

---

## The Tensor Framework

```
tensor-pretrain     train base models from scratch
tensor-adapt        extend base knowledge with nano adapters
tensor-inference    run base models + hot-swap adapters at runtime     ← you are here
```

Each tool is independent. Together they form a complete pipeline — from training
a base model to deploying a system that loads and unloads domain knowledge on demand.

---

## Why

Every major inference library is either Python-first, tied to a single hardware vendor,
or built around a single model family. `tensor-inference` is none of these.

- **Native C++** — no interpreter, no overhead, no runtime surprises
- **Any hardware** — CPU (x86, ARM), CUDA (NVIDIA), Metal (Apple Silicon + Neural Engine), Vulkan (cross-platform GPU)
- **Any model** — loads `.safetensors` and `.gguf` directly, no conversion needed
- **Adapter-native** — designed from the ground up to load and hot-swap `tensor-adapt` adapters at runtime
- **Any language** — stable C ABI lets Rust, Go, Zig, Swift, C#, or anything else call it directly
- **Portable** — compiles and runs anywhere C++ runs, including embedded and edge devices
- **Pre-tuned backends** — Metal and CUDA paths are optimized for their hardware, not generic wrappers

---

## What is a Transformer?

The transformer architecture introduced in *Attention is All You Need* (Vaswani et al., 2017)
is the foundation of every major AI model in production today. `tensor-inference` runs
all of them:

- **Decoder-only** — LLaMA, Qwen, Mistral, DeepSeek, Phi, Gemma, Falcon, Tensor Series
- **Encoder-only** — BERT, RoBERTa, DeBERTa, ModernBERT
- **Encoder-Decoder** — T5, BART, Whisper
- **Embedding models** — Jina, BGE, E5, Nomic
- **Vision Transformers** — ViT, SigLIP, CLIP
- **Multimodal** — LLaVA, Idefics, Qwen-VL, InternVL

The architecture is the scope. Not any single model family, not any single vendor.

---

## Adapter runtime

`tensor-inference` is the runtime half of `tensor-adapt`. Adapters trained with
`tensor-adapt` are first-class citizens here — not an afterthought.

The base model is loaded once and stays frozen in memory. Adapters are injected
per request based on context. When context changes, adapters swap in microseconds.
No model reload. No reallocation. The base never moves.

```
base model (loaded once, frozen)
    ├── adapter: github.com/gin-gonic/gin        inject → run → eject
    ├── adapter: github.com/sqlc-dev/sqlc        inject → run → eject
    ├── adapter: github.com/pressly/goose        inject → run → eject
    ├── adapter: github.com/riverqueue/river     inject → run → eject
    └── ... thousands more
```

Adapter validation is strict — `tensor-inference` reads the `adapter.json` produced
by `tensor-adapt` and rejects any adapter that was not trained against the loaded base.
Wrong base, wrong architecture, wrong layer range — all caught at load time, not at runtime.

---

## Structure

```
tensor-inference/
│
├── core/           — zero-dependency primitives
│                     DType, Shape, TensorView
│                     no platform headers, no compute, no allocation
│                     everything else builds on this
│
├── backend/        — CPU, CUDA, Metal, Vulkan
│                     swappable compute interface
│                     device memory allocation, kernel dispatch
│                     all platform headers live here and only here
│
├── parser/         — safetensors, GGUF, config.json, tokenizer.json
│                     memory-maps files, parses headers
│                     returns TensorView — never allocates weight memory
│                     no dependency on backend — no compute, no platform headers
│
├── tokenizer/      — text processing pipeline
│   ├── encoder/    — text → tokens (BPE, WordPiece, SentencePiece, tiktoken)
│   └── decoder/    — tokens → text, special token handling, chat templates
│
├── models/         — architecture definitions
│                     attention, MLP, RoPE, RMSNorm, MoE
│                     LlamaModel, QwenModel, BertModel, T5Model, ViT, Tensor Series
│                     KVCache lives here — shape is architecture-dependent
│
├── adapter/        — tensor-adapt adapter loading and hot-swap
│                     validates adapter.json against loaded base
│                     injects A/B matrices into target layers
│                     swap in microseconds — no reallocation
│
└── inference/      — generation loop, sampling, batching, streaming
```

**Dependencies flow one direction only:**

```
core ← backend ← models ← adapter ← inference
core ← parser  ← models
core ← tokenizer
```

`core` has no dependencies. `backend` builds on `core` — all platform headers stop here.
`parser` builds on `core` and never touches `backend` — parsing a file needs no GPU.
`models` is where `parser` and `backend` meet. `adapter` sits on top of `models` — it
needs to know the architecture to inject into the right layers. `inference` drives
everything and knows nothing about formats, devices, or kernels.

---

## End-to-end example

```cpp
#include <tensor/parser/weight_map.hpp>
#include <tensor/parser/config.hpp>
#include <tensor/backend/device.hpp>
#include <tensor/models/llama/llama_model.hpp>
#include <tensor/tokenizer/tokenizer.hpp>
#include <tensor/adapter/adapter.hpp>
#include <tensor/inference/generator.hpp>
#include <tensor/inference/sampling/top_p.hpp>

using tensor::backend::Device;
using tensor::models::llama::LlamaModel;
using tensor::tokenizer::Tokenizer;
using tensor::adapter::Adapter;
using tensor::inference::Generator;
using tensor::inference::GenerateOptions;
using tensor::inference::sampling::TopP;
using tensor::parser::WeightMap;
using tensor::parser::ModelConfig;

int main() {
    // 1. parse
    auto weights   = WeightMap::open("./models/Llama-3.1-8B/");
    auto config    = ModelConfig::from_file("./models/Llama-3.1-8B/config.json");
    auto tokenizer = Tokenizer::from_files("./models/Llama-3.1-8B/");

    // 2. pick a device
    auto device = Device::cuda(0);   // or Device::metal(), Device::cpu()

    // 3. load base model — frozen from this point forward
    auto model = LlamaModel::load(weights, config, device);

    // 4. load a tensor-adapt adapter — validated against base at load time
    auto adapter = Adapter::load("./adapters/golang-gin/", model);

    // 5. inject adapter — microseconds, no reallocation
    model.inject(adapter);

    // 6. generate
    auto gen = Generator::create(model, tokenizer);

    auto options = GenerateOptions {
        .max_new_tokens = 512,
        .sampler        = TopP { .temperature = 0.8f, .p = 0.95f },
    };

    auto prompt = tokenizer.apply_chat_template({
        { "system", "You are a Go expert." },
        { "user",   "Show me a gin route with middleware." },
    });

    auto result = gen.generate(tokenizer.encode(prompt), options);
    std::cout << tokenizer.decode(result.output_ids) << "\n";

    // 7. swap adapter — eject gin, inject sqlc
    model.eject(adapter);
    auto adapter2 = Adapter::load("./adapters/golang-sqlc/", model);
    model.inject(adapter2);

    // same base, new knowledge, zero reload
}
```

---

## Adapter loading

```cpp
#include <tensor/adapter/adapter.hpp>

using tensor::adapter::Adapter;
```

```cpp
// load and validate — reads adapter.json, checks base model ref
auto adapter = Adapter::load("./adapters/golang-gin/", model);

// inspect
std::string domain     = adapter.domain();      // "golang/gin"
std::string base_ref   = adapter.base_model();  // "meta-llama/Llama-3.1-8B"
int         rank       = adapter.rank();         // 2
float       alpha      = adapter.alpha();        // 2.0

// inject into base — modifies forward pass, not base weights
model.inject(adapter);

// eject — restores clean base forward pass
model.eject(adapter);

// or scope-based — auto ejects when scope exits
{
    auto scope = model.scoped_inject(adapter);
    // adapter active here
}
// adapter ejected here
```

Validation is strict. `Adapter::load` throws if:
- `adapter.json` is missing or malformed
- `base_model` in `adapter.json` does not match the loaded model
- Layer range in adapter exceeds model depth
- Adapter weights are the wrong shape for the target layers

---

## Parser

`parser` reads model files. It builds on `core` and nothing else — linking it
brings in no compute stack, no platform headers. Weight data is memory-mapped.
`TensorView` points into the mapped region. Bytes are never copied at parse time.

```cpp
// open any supported format from a directory — detects automatically
auto weights = WeightMap::open("./models/Llama-3.1-8B/");

// format-explicit
auto from_st   = WeightMap::from_safetensors("./models/Llama-3.1-8B/");
auto from_gguf = WeightMap::from_gguf("./models/Llama-3.1-8B-Q4_K_M.gguf");

// sharded models — transparent, same API
auto sharded = WeightMap::open("./models/Llama-3.1-70B/");

// tensor access
TensorView embed = weights.tensor("model.embed_tokens.weight");
bool       exists = weights.contains("model.layers.0.self_attn.q_proj.weight");
std::size_t count = weights.size();
```

Detection order: sharded `.safetensors` index first, single `.safetensors` second,
`.gguf` third. Explicit calls bypass detection entirely.

---

## Backend

`backend` owns device memory and kernel dispatch. All platform headers — CUDA, Metal,
Vulkan — live here and never leak upward. Everything above calls through the
`backend::Device` and `backend::Tensor` interface.

```cpp
auto cpu    = Device::cpu();      // always available
auto cuda0  = Device::cuda(0);   // NVIDIA GPU
auto metal  = Device::metal();   // Apple Silicon
auto vulkan = Device::vulkan(0); // cross-platform

// upload host → device
Tensor t = device.upload(view);           // synchronous
Tensor t = device.upload(view, stream);   // async

// metadata — no transfer
DType       dtype  = t.dtype();
Shape       shape  = t.shape();
std::size_t nbytes = t.nbytes();
```

`backend::ops` is the internal kernel layer — model code calls it, user code never does.

---

## Models

```cpp
#include <tensor/models/llama/llama_model.hpp>
#include <tensor/models/qwen/qwen_model.hpp>
#include <tensor/models/bert/bert_model.hpp>
#include <tensor/models/t5/t5_model.hpp>
#include <tensor/models/vit/vit_model.hpp>
#include <tensor/models/tensor/tensor_model.hpp>   // native Tensor Series
```

```cpp
// decoder-only — LLaMA, Mistral, DeepSeek, Phi, Gemma, Falcon
auto model = LlamaModel::load(weights, config, device);

// Tensor Series native
auto model = TensorModel::load(weights, config, device);

// encoder-only
auto bert = BertModel::load(weights, config, device);

// encoder-decoder
auto t5 = T5Model::load(weights, config, device);
```

`inference::Generator` is generic over model architecture through a concept,
not inheritance. Any type satisfying `models::CausalLM` works — no vtable overhead.

---

## Tokenizer

```cpp
// load from model directory — detects algorithm automatically
auto tokenizer = Tokenizer::from_files("./models/Llama-3.1-8B/");

// encode / decode
std::vector<int32_t> ids  = tokenizer.encode("Hello, world!");
std::string          text = tokenizer.decode(ids);

// chat template
std::string prompt = tokenizer.apply_chat_template({
    { "system", "You are a helpful assistant." },
    { "user",   "Explain RoPE embeddings." },
});
```

---

## Inference

```cpp
auto gen = Generator::create(model, tokenizer);

// standard
auto result = gen.generate(input_ids, {
    .max_new_tokens = 512,
    .sampler        = TopP { .temperature = 0.8f, .p = 0.95f },
    .stop_strings   = { "<|eot_id|>" },
});

// streaming
auto result = gen.generate(input_ids, {
    .max_new_tokens = 512,
    .sampler        = TopP { .temperature = 0.8f, .p = 0.95f },
    .on_token       = [&](int32_t id) {
        std::cout << tokenizer.decode({ id }) << std::flush;
    },
});
```

### Samplers

```
Greedy      — deterministic, highest probability token
TopP        — nucleus sampling, cumulative probability threshold
TopK        — sample from k highest probability tokens
MinP        — exclude tokens below min_p × max token probability
BeamSearch  — maintain n beams, return highest scoring sequence
```

---

## Supported formats

```
safetensors  — default PyTorch/HuggingFace export, memory-mappable
GGUF         — quantized models, llama.cpp compatible
```

---

## Supported hardware

```
CPU     — x86 and ARM, always available, reference backend
CUDA    — NVIDIA sm_70+ (Volta, Turing, Ampere, Hopper, Ada)
Metal   — Apple Silicon, Metal Performance Shaders + Neural Engine
Vulkan  — any Vulkan 1.3-capable GPU, cross-platform
```

---

## Build

### Requirements

- CMake 3.21+
- C++20 compiler (GCC 11+, Clang 14+, MSVC 19.30+)
- CUDA Toolkit 11.8+ for CUDA backend (optional)
- Xcode 14+ for Metal backend (optional)

### Build

```bash
git clone https://github.com/netangular/tensor-inference
cd tensor-inference

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### CMake options

| Option | Default | Description |
|---|---|---|
| `TENSOR_BACKEND_CUDA` | `ON` if CUDA found | Enable CUDA backend |
| `TENSOR_BACKEND_METAL` | `ON` if on macOS | Enable Metal backend |
| `TENSOR_BACKEND_VULKAN` | `OFF` | Enable Vulkan backend |
| `TENSOR_BACKEND_CPU` | `ON` | CPU fallback, always built |
| `TENSOR_BUILD_TESTS` | `OFF` | Build tests |

---

## Status

```
core        ████████████████████  complete
parser      ████████████████████  complete
backend     ████████░░░░░░░░░░░░  in progress
models      ██████░░░░░░░░░░░░░░  in progress
tokenizer   ██████░░░░░░░░░░░░░░  in progress
adapter     ████░░░░░░░░░░░░░░░░  in progress
inference   ████░░░░░░░░░░░░░░░░  in progress
```

---

## Non-goals

- **Training** — use `tensor-pretrain`
- **Adapter training** — use `tensor-adapt`
- **GGUF or pickle output** — safetensors only for anything produced by this library
- **Python bindings** — C ABI covers every language that matters at this level

---

*Part of the **Tensor Framework** by Netangular.*
*tensor-pretrain → tensor-adapt → tensor-inference*
*Apache 2.0 — free to use, modify, and build on, including for commercial purposes.*