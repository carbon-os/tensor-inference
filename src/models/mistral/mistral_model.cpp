#include <tensor/models/mistral/mistral_model.hpp>
#include <tensor/backend/ops.hpp>

#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

namespace ops = tensor::backend::ops;
using json = nlohmann::json;

namespace tensor::models::mistral {

static backend::Tensor load_w(const parser::WeightMap& wm, const std::string& name, const backend::Device& device) {
    return backend::Tensor::from_view(wm.tensor(name), device);
}

MistralModel MistralModel::load(const parser::WeightMap& weights, const std::string& model_dir, const backend::Device& device) {
    MistralModel m;
    m.device_ = device;
    device.make_current();

    // 1. Custom JSON parsing to handle multimodal nested text_config
    std::ifstream f(model_dir + "/config.json");
    json j = json::parse(f);
    
    json t_cfg = j.contains("text_config") ? j["text_config"] : j;

    m.cfg_.vocab_size          = t_cfg.value("vocab_size", 32000);
    m.cfg_.hidden_size         = t_cfg.value("hidden_size", 4096);
    m.cfg_.intermediate_size   = t_cfg.value("intermediate_size", 14336);
    m.cfg_.num_hidden_layers   = t_cfg.value("num_hidden_layers", 32);
    m.cfg_.num_attention_heads = t_cfg.value("num_attention_heads", 32);
    m.cfg_.num_key_value_heads = t_cfg.value("num_key_value_heads", 8);
    m.cfg_.rms_norm_eps        = t_cfg.value("rms_norm_eps", 1e-5f);
    
    if (t_cfg.contains("head_dim")) {
        m.cfg_.head_dim_ = t_cfg["head_dim"];
    } else {
        m.cfg_.head_dim_ = m.cfg_.hidden_size / m.cfg_.num_attention_heads;
    }

    if (t_cfg.contains("sliding_window") && !t_cfg["sliding_window"].is_null()) {
        m.cfg_.sliding_window = t_cfg["sliding_window"];
    }

    m.cfg_.bos_token_id = j.value("bos_token_id", 1);
    m.cfg_.eos_token_id = j.value("eos_token_id", 2);

    // RoPE YaRN / Scaling parameters
    if (t_cfg.contains("rope_parameters")) {
        auto rope = t_cfg["rope_parameters"];
        m.cfg_.rope_theta = rope.value("rope_theta", 1000000.0f);
        if (rope.value("type", "") == "yarn" || rope.value("rope_type", "") == "yarn") {
            m.cfg_.rope_scaling.apply_scaling = true;
            m.cfg_.rope_scaling.factor = rope.value("factor", 16.0f);
            m.cfg_.rope_scaling.original_max_pos = rope.value("original_max_position_embeddings", 16384);
        }
    } else {
        m.cfg_.rope_theta = t_cfg.value("rope_theta", 10000.0f);
    }

    std::cout << "[mistral] loading " << m.cfg_.num_hidden_layers << " layers\n";
    if (m.cfg_.sliding_window > 0) std::cout << "[mistral] sliding window: " << m.cfg_.sliding_window << "\n";

    // 2. Discover if weights are nested under "language_model." or "text_model."
    std::string prefix = "";
    if (weights.contains("language_model.model.embed_tokens.weight")) {
        prefix = "language_model.";
    } else if (weights.contains("text_model.model.embed_tokens.weight")) {
        prefix = "text_model.";
    }
    
    if (prefix != "") std::cout << "[mistral] multimodal text prefix detected: " << prefix << "\n";

    m.embed_tokens_ = load_w(weights, prefix + "model.embed_tokens.weight", device);
    m.norm_         = load_w(weights, prefix + "model.norm.weight", device);

    if (weights.contains(prefix + "lm_head.weight")) {
        m.lm_head_ = load_w(weights, prefix + "lm_head.weight", device);
    } else {
        m.lm_head_ = backend::Tensor::from_view(weights.tensor(prefix + "model.embed_tokens.weight"), device);
    }

    m.layers_.resize(m.cfg_.num_hidden_layers);
    for (std::size_t i = 0; i < m.cfg_.num_hidden_layers; ++i) {
        const std::string p = prefix + "model.layers." + std::to_string(i) + ".";
        MistralLayer& l = m.layers_[i];

        l.input_layernorm          = load_w(weights, p + "input_layernorm.weight", device);
        l.q_proj                   = load_w(weights, p + "self_attn.q_proj.weight", device);
        l.k_proj                   = load_w(weights, p + "self_attn.k_proj.weight", device);
        l.v_proj                   = load_w(weights, p + "self_attn.v_proj.weight", device);
        l.o_proj                   = load_w(weights, p + "self_attn.o_proj.weight", device);
        l.post_attention_layernorm = load_w(weights, p + "post_attention_layernorm.weight", device);
        l.gate_proj                = load_w(weights, p + "mlp.gate_proj.weight", device);
        l.up_proj                  = load_w(weights, p + "mlp.up_proj.weight", device);
        l.down_proj                = load_w(weights, p + "mlp.down_proj.weight", device);
    }

    std::cout << "[mistral] all weights loaded\n";
    return m;
}

KVCache MistralModel::make_cache(std::size_t max_seq_len) const {
    return KVCache(cfg_.num_hidden_layers, cfg_.num_key_value_heads, cfg_.head_dim(), max_seq_len, backend::DType::BF16, device_);
}

backend::Tensor MistralModel::layer_forward(const backend::Tensor& hidden, MistralLayer& layer, KVCache::LayerCache& lc, int seq_offset) {
    const int seq_len  = static_cast<int>(hidden.shape()[0]);
    const int n_heads  = static_cast<int>(cfg_.num_attention_heads);
    const int n_kv     = static_cast<int>(cfg_.num_key_value_heads);
    const int head_dim = static_cast<int>(cfg_.head_dim());
    const int hidden_s = static_cast<int>(cfg_.hidden_size);

    backend::Tensor normed = ops::rms_norm(hidden, layer.input_layernorm, cfg_.rms_norm_eps);

    backend::Tensor q = ops::matmul_t(normed, layer.q_proj);
    backend::Tensor k = ops::matmul_t(normed, layer.k_proj);
    backend::Tensor v = ops::matmul_t(normed, layer.v_proj);

    q = ops::reshape(q, {static_cast<std::size_t>(seq_len), static_cast<std::size_t>(n_heads), static_cast<std::size_t>(head_dim)});
    k = ops::reshape(k, {static_cast<std::size_t>(seq_len), static_cast<std::size_t>(n_kv),    static_cast<std::size_t>(head_dim)});
    v = ops::reshape(v, {static_cast<std::size_t>(seq_len), static_cast<std::size_t>(n_kv),    static_cast<std::size_t>(head_dim)});

    q = ops::rope(q, seq_offset, cfg_.rope_theta, cfg_.rope_scaling);
    k = ops::rope(k, seq_offset, cfg_.rope_theta, cfg_.rope_scaling);

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
    
    // Mistral specific: Pass the Sliding Window parameter
    backend::Tensor attn_out = ops::attention(q, k_ctx, v_ctx, causal, cfg_.sliding_window);
    
    // FIXED: Reshape to n_heads * head_dim, as the projection may be wider than hidden_size
    attn_out = ops::reshape(attn_out, {static_cast<std::size_t>(seq_len), static_cast<std::size_t>(n_heads * head_dim)});

    backend::Tensor attn_proj = ops::matmul_t(attn_out, layer.o_proj);
    backend::Tensor residual1 = ops::add(hidden, attn_proj);

    backend::Tensor normed2 = ops::rms_norm(residual1, layer.post_attention_layernorm, cfg_.rms_norm_eps);

    backend::Tensor gate = ops::matmul_t(normed2, layer.gate_proj);
    backend::Tensor up   = ops::matmul_t(normed2, layer.up_proj);

    backend::Tensor act  = ops::mul(ops::silu(gate), up);
    backend::Tensor down = ops::matmul_t(act, layer.down_proj);

    return ops::add(residual1, down);
}

backend::Tensor MistralModel::forward(const std::vector<int32_t>& input_ids, KVCache& cache) {
    device_.make_current();
    const int seq_offset = static_cast<int>(cache.seq_len());

    backend::Tensor hidden = ops::embedding(embed_tokens_, input_ids);

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        hidden = layer_forward(hidden, layers_[i], cache.layer(i), seq_offset);
    }

    hidden = ops::rms_norm(hidden, norm_, cfg_.rms_norm_eps);
    backend::Tensor logits = ops::matmul_t(hidden, lm_head_);

    cache.advance(input_ids.size());
    return logits;
}

} // namespace tensor::models::mistral