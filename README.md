```markdown
# tensor-inference

> Part of the **Tensor Framework** by Netangular

A native C++ inference engine for transformer-architecture models.
Load any model, run on any hardware, hot-swap knowledge adapters at runtime.

---

## The Tensor Framework

```
tensor-pretrain     train base models from scratch
tensor-adapt        extend base knowledge with tensor adapters
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
- **Router-native** — built from the ground up to semantically route and hot-swap tensor adapters at runtime
- **Any language** — stable C ABI lets Rust, Go, Zig, Swift, C#, or anything else call it directly
- **Portable** — compiles and runs anywhere C++ runs, including embedded and edge devices

---

## Tensor Adapters and the Native Router

This is what separates `tensor-inference` from a standard inference engine.

### What a Tensor Adapter is

A tensor adapter is a low-rank weight delta trained against a frozen base model
using `tensor-adapt`. Structurally it is LoRA-compatible — A/B matrices injected
into target layers, summed with frozen base activations at inference time. The base
never moves. The adapter adds a focused residual on top.

But a tensor adapter is not a general-purpose LoRA file. It carries a second
artifact alongside its weights: a **Semantic Centroid** — a 512-dimensional vector
computed automatically during training that describes where in the model's latent
space that adapter's knowledge lives. Not a label you wrote. Not an embedding of
the domain name. A mathematical fingerprint of what the gradient signal said the
adapter actually learned.

That centroid is what makes the adapter discoverable at runtime without any manual
routing rules.

### What the Router does

`tensor-inference` builds a **Product Key Memory (PKM) index** at startup by
reading the `.centroid` file from every adapter in the adapter directory. Each
centroid is quantized into a discrete address in a 512×512 = 262,144 bucket space
using two learned codebooks.

At inference time, the router reads the base model's current residual stream —
the same vector space the centroids live in — and finds the nearest PKM address
in O(1). The adapter registered at that address is fetched from RAM into VRAM if
it isn't already there, and injected into the forward pass.

No embedding call. No keyword lookup. No manual routing configuration.
The base model's own internal state is the search query. The router answers it
before the next token is generated.

### The goal: memory extension, not capability injection

This is an important distinction.

Tensor adapters are not trying to make the base model smarter. They are not
adding new reasoning patterns or retraining fundamental capabilities. The base
model already has those — baked in during pretraining at a scale no adapter
could replicate.

What large-scale pretraining takes away is **specificity**. A model trained on
the entire internet learns everything broadly and nothing deeply. Specific library
APIs, niche domain conventions, internal codebases, specialized terminology —
these are statistically overwhelmed by the sheer volume of general data. The
patterns are still there in the base weights. The model understands the shape of
the knowledge. It just lost the precise surface during overtraining on the general
corpus.

Tensor adapters restore that surface. Each adapter is trained on a narrow,
focused slice — a single library, a single domain, a single codebase. It adds
back the specific residual signal the base lost, in the exact layer positions
where the base already has the underlying pattern to work with.

The result is a base model that behaves as though it has deep familiarity with
whatever domain the current adapter covers — not because it was retrained, but
because the missing specificity was injected back in, on demand, for exactly
as long as it's needed.

One base. Thousands of knowledge domains. Each one a few megabytes.
Each one swapped in microseconds. Each one semantically discoverable
without any human-written routing rules.

---

## What is a Transformer?

The transformer architecture introduced in *Attention is All You Need*
(Vaswani et al., 2017) is the foundation of every major AI model in production
today. `tensor-inference` runs all of them:

- **Decoder-only** — LLaMA, Qwen, Mistral, DeepSeek, Phi, Gemma, Falcon, Tensor Series
- **Encoder-only** — BERT, RoBERTa, DeBERTa, ModernBERT
- **Encoder-Decoder** — T5, BART, Whisper
- **Embedding models** — Jina, BGE, E5, Nomic
- **Vision Transformers** — ViT, SigLIP, CLIP
- **Multimodal** — LLaVA, Idefics, Qwen-VL, InternVL

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
├── adapter/        — single adapter unit: load, validate, inject, eject
│                     validates adapter.json against loaded base at load time
│                     injects A/B matrices into target layers
│                     ejects cleanly — base forward pass fully restored
│                     knows nothing about routing or other adapters
│
├── router/         — semantic adapter routing at runtime
│   │
│   ├── index/      — PKM index built at startup from .centroid files
│   │                 loads two 512-key codebooks
│   │                 quantizes each centroid into a discrete (i,j) address
│   │                 registry: address → [adapter_id, ...]
│   │                 read-only after startup, lockfree at query time
│   │
│   ├── query/      — runtime lookup
│   │                 residual stream → nearest PKM address → ranked adapter IDs
│   │                 dot product against both codebook halves
│   │                 O(1) per token — no scan, no embedding call
│   │
│   ├── pool/       — adapter lifecycle and VRAM budget
│   │                 tracks which adapters are resident in VRAM
│   │                 LRU eviction when budget is exceeded
│   │                 async prefetch from RAM when a candidate is identified
│   │                 calls into adapter/ for the actual inject and eject
│   │
│   └── scheduler/  — swap timing and stability
│                     tracks semantic shift in the residual stream across tokens
│                     triggers pool/ prefetch ahead of a confirmed domain shift
│                     hysteresis threshold prevents flicker at ambiguous boundaries
│                     holds current adapter until shift is decisive
│
└── inference/      — generation loop, sampling, batching, streaming
                      calls router/ for adapter decisions
                      never touches adapter/ directly
```

**Dependencies flow one direction only:**

```
core ← backend ← models ← adapter ← router ← inference
core ← parser  ← models
core ← tokenizer
```

`adapter/` handles one adapter in isolation — load, validate, inject, eject.
`router/` manages all adapters collectively — index, pool, swap timing.
`router/` calls down into `adapter/`. `inference/` calls `router/`.
Nothing flows the other direction.

---

## Adapter runtime

The base model is loaded once and stays frozen in memory. The router indexes
all available adapters at startup from their `.centroid` files. From that point
forward, adapter selection is fully automatic — driven by the base model's own
residual stream, not by any external signal.

```
base model (loaded once, frozen)
    ├── adapter: golang/gin              auto-routed → inject → run → eject
    ├── adapter: golang/sqlc             auto-routed → inject → run → eject
    ├── adapter: rust/tokio              auto-routed → inject → run → eject
    ├── adapter: internal/payments-api   auto-routed → inject → run → eject
    └── ... thousands more
```

Adapter validation is strict — `tensor-inference` reads the `adapter.json`
produced by `tensor-adapt` and rejects any adapter that was not trained against
the loaded base. Wrong base, wrong architecture, wrong layer range, missing
centroid — all caught at startup, not at runtime.

---

## End-to-end example

```cpp
#include <tensor/models/llama/llama_model.hpp>
#include <tensor/tokenizer/tokenizer.hpp>
#include <tensor/router/router.hpp>
#include <tensor/inference/generator.hpp>
#include <tensor/inference/sampling/top_p.hpp>

int main() {
    // 1. load frozen base
    auto weights   = WeightMap::open("./models/Llama-3.1-8B/");
    auto config    = ModelConfig::from_file("./models/Llama-3.1-8B/config.json");
    auto tokenizer = Tokenizer::from_files("./models/Llama-3.1-8B/");
    auto device    = Device::cuda(0);
    auto model     = LlamaModel::load(weights, config, device);

    // 2. build router — reads all .centroid files, builds PKM index
    auto router = Router::build("./adapters/", model);

    // 3. generate — router handles all adapter decisions automatically
    auto gen = Generator::create(model, tokenizer, router);

    auto prompt = tokenizer.apply_chat_template({
        { "system", "You are a Go expert." },
        { "user",   "Show me a gin route with middleware." },
    });

    auto result = gen.generate(tokenizer.encode(prompt), {
        .max_new_tokens = 512,
        .sampler        = TopP { .temperature = 0.8f, .p = 0.95f },
    });

    std::cout << tokenizer.decode(result.output_ids) << "\n";
}
```

The router is the only new piece. Everything else is unchanged.
Swap the prompt to a Rust question — the router routes to a different adapter.
No code change. No adapter name. No manual selection.

---

## Supported base models

```
Decoder-only     LLaMA 3 / 3.1 / 3.2 / 3.3
                 Qwen 2 / 2.5
                 Mistral / Mixtral
                 DeepSeek V2 / V3
                 Phi-3 / Phi-4
                 Gemma / Gemma 2
                 Falcon
                 Tensor Series (tensor-pretrain output)

Encoder-only     BERT / RoBERTa / DeBERTa / ModernBERT

Encoder-Decoder  T5 / BART / Whisper

Embedding        Jina / BGE / E5 / Nomic

Vision           ViT / SigLIP / CLIP

Multimodal       LLaVA / Idefics / Qwen-VL / InternVL
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
router      ████░░░░░░░░░░░░░░░░  in progress
inference   ████░░░░░░░░░░░░░░░░  in progress
```

---

## Non-goals

- **Training** — use `tensor-pretrain`
- **Adapter training** — use `tensor-adapt`
- **Making the base model smarter** — tensor adapters restore specificity, they do not add reasoning capability
- **Manual adapter selection** — the router exists so you never have to name an adapter at runtime
- **GGUF or pickle output** — safetensors only for anything produced by this library
- **Python bindings** — C ABI covers every language that matters at this level

---

*Part of the **Tensor Framework** by Netangular.*
*tensor-pretrain → tensor-adapt → tensor-inference*
*Apache 2.0 — free to use, modify, and build on, including for commercial purposes.*
```

The big additions are the **Tensor Adapters and the Native Router** section which carries the full philosophy — what adapters are, what the router does mechanically, and critically the memory extension framing rather than capability injection. That distinction also made it into Non-goals explicitly so it's hard to miss. The router directory structure slots cleanly between `adapter/` and `inference/` with the dependency chain updated to reflect it.