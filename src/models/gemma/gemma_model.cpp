#include <transformers/models/gemma/gemma_model.hpp>
#include <transformers/backend/ops.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

namespace ops = transformers::backend::ops;

namespace transformers::models::gemma {

static backend::Tensor load_weight(const parser::WeightMap& wm, const std::string& name, const backend::Device& device) {
    return backend::Tensor::from_view(wm.tensor(name), device);
}

GemmaModel GemmaModel::load(const parser::WeightMap& weights, const parser::ModelConfig& config, const backend::Device& device) {
    GemmaModel m;
    m.cfg_    = GemmaConfig::from_model_config(config);
    m.device_ = device;
    device.make_current();

    std::cout << "[gemma] loading " << m.cfg_.num_hidden_layers << " layers\n";

    m.embed_tokens_ = load_weight(weights, "model.embed_tokens.weight", device);
    m.norm_         = load_weight(weights, "model.norm.weight", device);

    if (weights.contains("lm_head.weight")) {
        m.lm_head_ = load_weight(weights, "lm_head.weight", device);
    } else {
        m.lm_head_ = backend::Tensor::from_view(weights.tensor("model.embed_tokens.weight"), device);
    }

    m.layers_.resize(m.cfg_.num_hidden_layers);
    for (std::size_t i = 0; i < m.cfg_.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        GemmaLayer& l = m.layers_[i];

        l.input_layernorm          = load_weight(weights, p + "input_layernorm.weight", device);
        l.q_proj                   = load_weight(weights, p + "self_attn.q_proj.weight", device);
        l.k_proj                   = load_weight(weights, p + "self_attn.k_proj.weight", device);
        l.v_proj                   = load_weight(weights, p + "self_attn.v_proj.weight", device);
        l.o_proj                   = load_weight(weights, p + "self_attn.o_proj.weight", device);
        l.post_attention_layernorm = load_weight(weights, p + "post_attention_layernorm.weight", device);
        l.gate_proj                = load_weight(weights, p + "mlp.gate_proj.weight", device);
        l.up_proj                  = load_weight(weights, p + "mlp.up_proj.weight", device);
        l.down_proj                = load_weight(weights, p + "mlp.down_proj.weight", device);
    }

    std::cout << "[gemma] all weights loaded\n";
    return m;
}

KVCache GemmaModel::make_cache(std::size_t max_seq_len) const {
    return KVCache(cfg_.num_hidden_layers, cfg_.num_key_value_heads, cfg_.head_dim(), max_seq_len, backend::DType::BF16, device_);
}

backend::Tensor GemmaModel::layer_forward(const backend::Tensor& hidden, GemmaLayer& layer, KVCache::LayerCache& lc, int seq_offset) {
    const int seq_len  = static_cast<int>(hidden.shape()[0]);
    const int n_heads  = static_cast<int>(cfg_.num_attention_heads);
    const int n_kv     = static_cast<int>(cfg_.num_key_value_heads);
    const int head_dim = static_cast<int>(cfg_.head_dim());
    const int hidden_s = static_cast<int>(cfg_.hidden_size);

    // Gemma uses the +1.0 offset norm
    backend::Tensor normed = ops::rms_norm_offset(hidden, layer.input_layernorm, cfg_.rms_norm_eps);

    backend::Tensor q = ops::matmul_t(normed, layer.q_proj);
    backend::Tensor k = ops::matmul_t(normed, layer.k_proj);
    backend::Tensor v = ops::matmul_t(normed, layer.v_proj);

    q = ops::reshape(q, {static_cast<std::size_t>(seq_len), static_cast<std::size_t>(n_heads), static_cast<std::size_t>(head_dim)});
    k = ops::reshape(k, {static_cast<std::size_t>(seq_len), static_cast<std::size_t>(n_kv),    static_cast<std::size_t>(head_dim)});
    v = ops::reshape(v, {static_cast<std::size_t>(seq_len), static_cast<std::size_t>(n_kv),    static_cast<std::size_t>(head_dim)});

    q = ops::rope(q, seq_offset, cfg_.rope_theta);
    k = ops::rope(k, seq_offset, cfg_.rope_theta);

    {
        const std::size_t kv_row_bytes = n_kv * head_dim * core::dtype_size(k.dtype());
        cudaMemcpy(static_cast<uint8_t*>(lc.keys.data_ptr()) + seq_offset * kv_row_bytes,
                   k.data_ptr(), k.nbytes(), cudaMemcpyDeviceToDevice);
        cudaMemcpy(static_cast<uint8_t*>(lc.values.data_ptr()) + seq_offset * kv_row_bytes,
                   v.data_ptr(), v.nbytes(), cudaMemcpyDeviceToDevice);
    }

    const std::size_t kv_len = static_cast<std::size_t>(seq_offset) + seq_len;
    backend::Tensor k_ctx = ops::slice(lc.keys,   0, 0, kv_len);
    backend::Tensor v_ctx = ops::slice(lc.values, 0, 0, kv_len);

    bool causal = (seq_len > 1);
    backend::Tensor attn_out = ops::attention(q, k_ctx, v_ctx, causal);
    attn_out = ops::reshape(attn_out, {static_cast<std::size_t>(seq_len), static_cast<std::size_t>(hidden_s)});

    backend::Tensor attn_proj = ops::matmul_t(attn_out, layer.o_proj);
    backend::Tensor residual1 = ops::add(hidden, attn_proj);

    // Gemma uses the +1.0 offset norm
    backend::Tensor normed2 = ops::rms_norm_offset(residual1, layer.post_attention_layernorm, cfg_.rms_norm_eps);

    backend::Tensor gate = ops::matmul_t(normed2, layer.gate_proj);
    backend::Tensor up   = ops::matmul_t(normed2, layer.up_proj);

    // GeGLU Activation
    backend::Tensor act  = ops::mul(ops::gelu(gate), up);
    backend::Tensor down = ops::matmul_t(act, layer.down_proj);

    return ops::add(residual1, down);
}

backend::Tensor GemmaModel::forward(const std::vector<int32_t>& input_ids, KVCache& cache) {
    device_.make_current();
    const int seq_offset = static_cast<int>(cache.seq_len());

    backend::Tensor hidden = ops::embedding(embed_tokens_, input_ids);

    // Gemma specific: Scale embeddings by sqrt(hidden_size)
    hidden = ops::mul(hidden, std::sqrt(static_cast<float>(cfg_.hidden_size)));

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        hidden = layer_forward(hidden, layers_[i], cache.layer(i), seq_offset);
    }

    // Gemma uses the +1.0 offset norm
    hidden = ops::rms_norm_offset(hidden, norm_, cfg_.rms_norm_eps);
    backend::Tensor logits = ops::matmul_t(hidden, lm_head_);

    cache.advance(input_ids.size());
    return logits;
}

} // namespace transformers::models::gemma