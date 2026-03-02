#pragma once

#include <transformers/models/llama/llama_config.hpp>
#include <transformers/models/kv_cache.hpp>
#include <transformers/backend/tensor.hpp>
#include <transformers/backend/device.hpp>
#include <transformers/parser/weight_map.hpp>
#include <transformers/parser/config.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace transformers::models::llama {

using backend::Device;
using backend::Tensor;

// All device-resident weights for one transformer layer.
struct LlamaLayer {
    Tensor input_layernorm;           // [hidden]
    Tensor q_proj;                    // [n_heads * head_dim, hidden]
    Tensor k_proj;                    // [n_kv_heads * head_dim, hidden]
    Tensor v_proj;                    // [n_kv_heads * head_dim, hidden]
    Tensor o_proj;                    // [hidden, n_heads * head_dim]
    Tensor post_attention_layernorm;  // [hidden]
    Tensor gate_proj;                 // [intermediate, hidden]
    Tensor up_proj;                   // [intermediate, hidden]
    Tensor down_proj;                 // [hidden, intermediate]
};

// ─────────────────────────────────────────────────────────────
//  LlamaModel — satisfies models::CausalLM
// ─────────────────────────────────────────────────────────────

class LlamaModel {
public:
    // Load all weights from a WeightMap onto device.
    static LlamaModel load(const parser::WeightMap& weights,
                           const parser::ModelConfig& config,
                           const Device& device);

    // ── CausalLM concept ────────────────────────────────────

    // input_ids: new tokens only (prompt on first call, one token on subsequent calls)
    // Returns logits: [seq_len, vocab_size]
    Tensor forward(const std::vector<int32_t>& input_ids, KVCache& cache);

    std::size_t vocab_size()   const noexcept { return cfg_.vocab_size; }
    int32_t     eos_token_id() const noexcept { return cfg_.eos_token_id; }

    // ── extras ──────────────────────────────────────────────

    const LlamaConfig& config() const noexcept { return cfg_; }
    const Device&      device() const noexcept { return device_; }

    KVCache make_cache(std::size_t max_seq_len) const;

    LlamaModel(LlamaModel&&) noexcept            = default;
    LlamaModel& operator=(LlamaModel&&) noexcept = default;
    LlamaModel(const LlamaModel&)                = delete;
    LlamaModel& operator=(const LlamaModel&)     = delete;

private:
    LlamaConfig             cfg_;
    Device                  device_;
    Tensor                  embed_tokens_;   // [vocab, hidden]
    std::vector<LlamaLayer> layers_;
    Tensor                  norm_;           // [hidden]
    Tensor                  lm_head_;        // [vocab, hidden]

    LlamaModel() = default;

    Tensor layer_forward(const Tensor& hidden,
                         LlamaLayer& layer,
                         KVCache::LayerCache& cache,
                         int seq_offset);
};

} // namespace transformers::models::llama