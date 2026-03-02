#include <transformers/models/llama/llama_model.hpp>
#include <transformers/backend/ops.hpp>
#include <transformers/parser/errors.hpp>

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace ops = transformers::backend::ops;
using transformers::parser::TensorNotFound;

namespace transformers::models::llama {

// ─────────────────────────────────────────────────────────────
//  Weight loading helpers
// ─────────────────────────────────────────────────────────────

static backend::Tensor load_weight(const parser::WeightMap& wm,
                                   const std::string& name,
                                   const Device& device) {
    // Throws TensorNotFound (from parser) if missing — intentional.
    auto view = wm.tensor(name);
    return backend::Tensor::from_view(view, device);
}

// ─────────────────────────────────────────────────────────────
//  LlamaModel::load
// ─────────────────────────────────────────────────────────────

LlamaModel LlamaModel::load(const parser::WeightMap& weights,
                             const parser::ModelConfig& config,
                             const Device& device) {
    LlamaModel m;
    m.cfg_    = LlamaConfig::from_model_config(config);
    m.device_ = device;
    device.make_current();

    std::cout << "[llama] loading " << m.cfg_.num_hidden_layers << " layers"
              << "  hidden=" << m.cfg_.hidden_size
              << "  heads=" << m.cfg_.num_attention_heads
              << "/" << m.cfg_.num_key_value_heads
              << "  vocab=" << m.cfg_.vocab_size
              << "\n";

    m.embed_tokens_ = load_weight(weights, "model.embed_tokens.weight", device);
    m.norm_         = load_weight(weights, "model.norm.weight", device);

    // lm_head may be tied to embed_tokens (weight tying)
    if (weights.contains("lm_head.weight")) {
        m.lm_head_ = load_weight(weights, "lm_head.weight", device);
    } else {
        // Weight-tied: lm_head == embed_tokens — upload a second copy
        auto view = weights.tensor("model.embed_tokens.weight");
        m.lm_head_ = backend::Tensor::from_view(view, device);
    }

    m.layers_.resize(m.cfg_.num_hidden_layers);
    for (std::size_t i = 0; i < m.cfg_.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        LlamaLayer& l = m.layers_[i];

        l.input_layernorm          = load_weight(weights, p + "input_layernorm.weight", device);
        l.q_proj                   = load_weight(weights, p + "self_attn.q_proj.weight", device);
        l.k_proj                   = load_weight(weights, p + "self_attn.k_proj.weight", device);
        l.v_proj                   = load_weight(weights, p + "self_attn.v_proj.weight", device);
        l.o_proj                   = load_weight(weights, p + "self_attn.o_proj.weight", device);
        l.post_attention_layernorm = load_weight(weights, p + "post_attention_layernorm.weight", device);
        l.gate_proj                = load_weight(weights, p + "mlp.gate_proj.weight", device);
        l.up_proj                  = load_weight(weights, p + "mlp.up_proj.weight", device);
        l.down_proj                = load_weight(weights, p + "mlp.down_proj.weight", device);

        if ((i + 1) % 8 == 0 || i == m.cfg_.num_hidden_layers - 1) {
            std::cout << "[llama] loaded layer " << (i + 1)
                      << "/" << m.cfg_.num_hidden_layers << "\n";
        }
    }

    std::cout << "[llama] all weights loaded\n";
    return m;
}

// ─────────────────────────────────────────────────────────────
//  make_cache
// ─────────────────────────────────────────────────────────────

KVCache LlamaModel::make_cache(std::size_t max_seq_len) const {
    return KVCache(
        cfg_.num_hidden_layers,
        cfg_.num_key_value_heads,
        cfg_.head_dim(),
        max_seq_len,
        backend::DType::BF16,
        device_
    );
}

// ─────────────────────────────────────────────────────────────
//  Single layer forward
// ─────────────────────────────────────────────────────────────

Tensor LlamaModel::layer_forward(const Tensor& hidden,
                                 LlamaLayer& layer,
                                 KVCache::LayerCache& lc,
                                 int seq_offset) {
    const int seq_len  = static_cast<int>(hidden.shape()[0]);
    const int n_heads  = static_cast<int>(cfg_.num_attention_heads);
    const int n_kv     = static_cast<int>(cfg_.num_key_value_heads);
    const int head_dim = static_cast<int>(cfg_.head_dim());
    const int hidden_s = static_cast<int>(cfg_.hidden_size);

    // ── pre-attention norm ────────────────────────────────────

    Tensor normed = ops::rms_norm(hidden, layer.input_layernorm, cfg_.rms_norm_eps);

    // ── QKV projections ───────────────────────────────────────

    // q: [seq_len, n_heads * head_dim]
    Tensor q = ops::matmul_t(normed, layer.q_proj);
    Tensor k = ops::matmul_t(normed, layer.k_proj);
    Tensor v = ops::matmul_t(normed, layer.v_proj);

    // Reshape to [seq_len, n_heads, head_dim]
    q = ops::reshape(q, {static_cast<std::size_t>(seq_len),
                         static_cast<std::size_t>(n_heads),
                         static_cast<std::size_t>(head_dim)});
    k = ops::reshape(k, {static_cast<std::size_t>(seq_len),
                         static_cast<std::size_t>(n_kv),
                         static_cast<std::size_t>(head_dim)});
    v = ops::reshape(v, {static_cast<std::size_t>(seq_len),
                         static_cast<std::size_t>(n_kv),
                         static_cast<std::size_t>(head_dim)});

    // ── RoPE ─────────────────────────────────────────────────
    //
    // Build the ops::RopeScaling descriptor from the model config so the
    // Llama-3 per-dimension frequency scaling is applied correctly.

    ops::RopeScaling rope_sc;
    rope_sc.apply_scaling    = cfg_.rope_scaling.enabled;
    rope_sc.factor           = cfg_.rope_scaling.factor;
    rope_sc.low_freq_factor  = cfg_.rope_scaling.low_freq_factor;
    rope_sc.high_freq_factor = cfg_.rope_scaling.high_freq_factor;
    rope_sc.original_max_pos = cfg_.rope_scaling.original_max_pos;

    q = ops::rope(q, seq_offset, cfg_.rope_theta, rope_sc);
    k = ops::rope(k, seq_offset, cfg_.rope_theta, rope_sc);

    // ── KV cache write ────────────────────────────────────────
    //
    // Write new K and V into the pre-allocated cache slots,
    // then pass the full [0..seq_offset+seq_len) slice to attention.

    {
        // Copy new K into cache at [seq_offset : seq_offset+seq_len]
        const std::size_t kv_row_bytes = n_kv * head_dim
                                       * core::dtype_size(k.dtype());
        cudaMemcpy(
            static_cast<uint8_t*>(lc.keys.data_ptr())
                + seq_offset * kv_row_bytes,
            k.data_ptr(), k.nbytes(),
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpy(
            static_cast<uint8_t*>(lc.values.data_ptr())
                + seq_offset * kv_row_bytes,
            v.data_ptr(), v.nbytes(),
            cudaMemcpyDeviceToDevice
        );
    }

    const std::size_t kv_len = static_cast<std::size_t>(seq_offset) + seq_len;

    // Slice the filled portion of the cache: [0, kv_len)
    Tensor k_ctx = ops::slice(lc.keys,   0, 0, kv_len);
    Tensor v_ctx = ops::slice(lc.values, 0, 0, kv_len);

    // ── attention ─────────────────────────────────────────────

    bool causal = (seq_len > 1);   // prefill = causal mask; decode = no mask needed
    Tensor attn_out = ops::attention(q, k_ctx, v_ctx, causal);
    // [seq_len, n_heads, head_dim] → [seq_len, hidden]
    attn_out = ops::reshape(attn_out, {static_cast<std::size_t>(seq_len),
                                       static_cast<std::size_t>(hidden_s)});

    // ── output projection + residual ──────────────────────────

    Tensor attn_proj = ops::matmul_t(attn_out, layer.o_proj);
    Tensor residual1 = ops::add(hidden, attn_proj);

    // ── FFN (SwiGLU) ──────────────────────────────────────────

    Tensor normed2 = ops::rms_norm(residual1, layer.post_attention_layernorm,
                                   cfg_.rms_norm_eps);

    Tensor gate = ops::matmul_t(normed2, layer.gate_proj);
    Tensor up   = ops::matmul_t(normed2, layer.up_proj);

    Tensor act  = ops::mul(ops::silu(gate), up);   // SwiGLU
    Tensor down = ops::matmul_t(act, layer.down_proj);

    return ops::add(residual1, down);
}

// ─────────────────────────────────────────────────────────────
//  LlamaModel::forward
// ─────────────────────────────────────────────────────────────

Tensor LlamaModel::forward(const std::vector<int32_t>& input_ids,
                            KVCache& cache) {
    if (input_ids.empty()) {
        throw std::invalid_argument("LlamaModel::forward: empty input_ids");
    }

    device_.make_current();

    const int seq_offset = static_cast<int>(cache.seq_len());

    // Token embeddings: [seq_len, hidden]
    Tensor hidden = ops::embedding(embed_tokens_, input_ids);

    // Transformer layers
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        hidden = layer_forward(hidden, layers_[i], cache.layer(i), seq_offset);
    }

    // Final norm
    hidden = ops::rms_norm(hidden, norm_, cfg_.rms_norm_eps);

    // LM head: [seq_len, vocab]
    Tensor logits = ops::matmul_t(hidden, lm_head_);

    // Advance cache position
    cache.advance(input_ids.size());

    return logits;
}

} // namespace transformers::models::llama