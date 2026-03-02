#pragma once

#include <transformers/backend/tensor.hpp>
#include <transformers/models/kv_cache.hpp>
#include <transformers/parser/config.hpp>
#include <transformers/parser/weight_map.hpp>

#include <vector>

namespace transformers::models::gemma {

struct GemmaConfig {
    std::size_t vocab_size              = 256000;
    std::size_t hidden_size             = 2048;
    std::size_t intermediate_size       = 16384;
    std::size_t num_hidden_layers       = 18;
    std::size_t num_attention_heads     = 8;
    std::size_t num_key_value_heads     = 1;
    std::size_t head_dim_               = 256;
    float       rms_norm_eps            = 1e-6f;
    float       rope_theta              = 10000.0f;
    int32_t     eos_token_id            = 1;
    
    std::size_t head_dim() const { return head_dim_; }

    static GemmaConfig from_model_config(const parser::ModelConfig& cfg) {
        GemmaConfig c;
        c.vocab_size              = cfg.vocab_size();
        c.hidden_size             = cfg.hidden_size();
        c.intermediate_size       = cfg.intermediate_size();
        c.num_hidden_layers       = cfg.num_hidden_layers();
        c.num_attention_heads     = cfg.num_attention_heads();
        c.num_key_value_heads     = cfg.num_key_value_heads();
        
        // Compute head_dim mathematically instead of using the raw JSON accessor
        c.head_dim_               = cfg.hidden_size() / cfg.num_attention_heads();
        
        c.rms_norm_eps            = cfg.rms_norm_eps();
        c.rope_theta              = cfg.rope_theta();
        c.eos_token_id            = cfg.eos_token_id();
        return c;
    }
};

struct GemmaLayer {
    backend::Tensor input_layernorm;
    backend::Tensor q_proj;
    backend::Tensor k_proj;
    backend::Tensor v_proj;
    backend::Tensor o_proj;
    backend::Tensor post_attention_layernorm;
    backend::Tensor gate_proj;
    backend::Tensor up_proj;
    backend::Tensor down_proj;
};

class GemmaModel {
public:
    static GemmaModel load(const parser::WeightMap& weights,
                           const parser::ModelConfig& config,
                           const backend::Device& device);

    backend::Tensor forward(const std::vector<int32_t>& input_ids, KVCache& cache);

    KVCache make_cache(std::size_t max_seq_len) const;

    std::size_t vocab_size()   const noexcept { return cfg_.vocab_size; }
    int32_t     eos_token_id() const noexcept { return cfg_.eos_token_id; }

private:
    backend::Tensor layer_forward(const backend::Tensor& hidden,
                                  GemmaLayer& layer,
                                  KVCache::LayerCache& lc,
                                  int seq_offset);

    GemmaConfig             cfg_;
    backend::Device         device_ = backend::Device::cpu();
    backend::Tensor         embed_tokens_;
    backend::Tensor         norm_;
    backend::Tensor         lm_head_;
    std::vector<GemmaLayer> layers_;
};

} // namespace transformers::models::gemma