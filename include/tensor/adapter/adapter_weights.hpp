#pragma once

#include <tensor/backend/tensor.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace tensor::adapter {

// ─────────────────────────────────────────────────────────────
//  A single low-rank projection pair
//
//  lora_A : [rank, in_features]   — kaiming-initialised
//  lora_B : [out_features, rank]  — zero-initialised
//
//  delta  = (x @ lora_A^T) @ lora_B^T   shape [seq, out_features]
//         = matmul_t(matmul_t(x, lora_A), lora_B)
// ─────────────────────────────────────────────────────────────

struct LoraProjection {
    backend::Tensor lora_A;   // [rank, in_features]
    backend::Tensor lora_B;   // [out_features, rank]
};

// ─────────────────────────────────────────────────────────────
//  Per-layer adapter weights
//  Each field is optional — only the projections that were
//  actually injected during training will be present.
// ─────────────────────────────────────────────────────────────

struct AdapterLayer {
    std::optional<LoraProjection> q_proj;
    std::optional<LoraProjection> k_proj;
    std::optional<LoraProjection> v_proj;
    std::optional<LoraProjection> o_proj;
    std::optional<LoraProjection> gate_proj;
    std::optional<LoraProjection> up_proj;
    std::optional<LoraProjection> down_proj;
};

// ─────────────────────────────────────────────────────────────
//  Metadata parsed from adapter.json
//  This is the contract written by tensor-adapt and read here.
// ─────────────────────────────────────────────────────────────

struct AdapterMeta {
    std::string domain;
    std::string architecture;   // "qwen2", "llama", …
    std::string base_model;     // e.g. "Qwen/Qwen2.5-0.5B"
    std::string base_sha;

    int         rank;
    float       alpha;

    std::size_t target_begin;   // first injected transformer layer
    std::size_t target_end;     // last  injected transformer layer (inclusive)

    bool inject_q;
    bool inject_k;
    bool inject_v;
    bool inject_o;
    bool inject_up;
    bool inject_down;

    std::size_t centroid_count; // informational only at this layer

    std::string tensor_adapt_version;
};

// ─────────────────────────────────────────────────────────────
//  Complete adapter — metadata + per-layer weight pairs
//
//  layers is indexed by transformer layer index.
//  Layers outside [target_begin, target_end] are default-
//  constructed (all optionals empty) and never touched.
// ─────────────────────────────────────────────────────────────

struct AdapterWeights {
    AdapterMeta               meta;
    std::vector<AdapterLayer> layers;   // size == num_hidden_layers of base

    // LoRA scaling factor applied to every delta: alpha / rank
    float scale() const noexcept {
        return meta.alpha / static_cast<float>(meta.rank);
    }
};

} // namespace tensor::adapter