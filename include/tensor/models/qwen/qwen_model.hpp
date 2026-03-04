#pragma once

#include <tensor/backend/tensor.hpp>
#include <tensor/models/kv_cache.hpp>
#include <tensor/parser/config.hpp>
#include <tensor/parser/weight_map.hpp>

#include <vector>

namespace tensor::models::qwen {

struct QwenConfig {
    std::size_t vocab_size              = 151936;
    std::size_t hidden_size             = 1536;
    std::size_t intermediate_size       = 8960;
    std::size_t num_hidden_layers       = 28;
    std::size_t num_attention_heads     = 12;
    std::size_t num_key_value_heads     = 2;
    std::size_t max_position_embeddings = 32768;
    float       rms_norm_eps            = 1e-6f;
    float       rope_theta              = 1000000.0f;
    int32_t     eos_token_id            = 151645;
    
    std::size_t head_dim() const { return hidden_size / num_attention_heads; }

    static QwenConfig from_model_config(const parser::ModelConfig& cfg) {
        QwenConfig c;
        c.vocab_size              = cfg.vocab_size();
        c.hidden_size             = cfg.hidden_size();
        c.intermediate_size       = cfg.intermediate_size();
        c.num_hidden_layers       = cfg.num_hidden_layers();
        c.num_attention_heads     = cfg.num_attention_heads();
        c.num_key_value_heads     = cfg.num_key_value_heads();
        c.max_position_embeddings = cfg.max_position_embeddings();
        c.rms_norm_eps            = cfg.rms_norm_eps();
        c.rope_theta              = cfg.rope_theta();
        c.eos_token_id            = cfg.eos_token_id();
        return c;
    }
};

struct QwenLayer {
    backend::Tensor input_layernorm;
    backend::Tensor q_proj_weight;
    backend::Tensor q_proj_bias;
    backend::Tensor k_proj_weight;
    backend::Tensor k_proj_bias;
    backend::Tensor v_proj_weight;
    backend::Tensor v_proj_bias;
    backend::Tensor o_proj_weight;
    backend::Tensor post_attention_layernorm;
    backend::Tensor gate_proj_weight;
    backend::Tensor up_proj_weight;
    backend::Tensor down_proj_weight;
};

class QwenModel {
public:
    static QwenModel load(const parser::WeightMap& weights,
                          const parser::ModelConfig& config,
                          const backend::Device& device);

    backend::Tensor forward(const std::vector<int32_t>& input_ids,
                            KVCache& cache);

    KVCache make_cache(std::size_t max_seq_len) const;

    std::size_t vocab_size()   const noexcept { return cfg_.vocab_size; }
    int32_t     eos_token_id() const noexcept { return cfg_.eos_token_id; }

private:
    backend::Tensor layer_forward(const backend::Tensor& hidden,
                                  QwenLayer& layer,
                                  KVCache::LayerCache& lc,
                                  int seq_offset);

    QwenConfig             cfg_;
    backend::Device        device_ = backend::Device::cpu();
    backend::Tensor        embed_tokens_;
    backend::Tensor        norm_;
    backend::Tensor        lm_head_;
    std::vector<QwenLayer> layers_;
};

} // namespace tensor::models::qwen