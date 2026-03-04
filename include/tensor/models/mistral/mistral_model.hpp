#pragma once

#include <tensor/backend/tensor.hpp>
#include <tensor/backend/ops.hpp>
#include <tensor/models/kv_cache.hpp>
#include <tensor/parser/weight_map.hpp>

#include <string>
#include <vector>

namespace tensor::models::mistral {

struct MistralConfig {
    std::size_t vocab_size              = 32000;
    std::size_t hidden_size             = 4096;
    std::size_t intermediate_size       = 14336;
    std::size_t num_hidden_layers       = 32;
    std::size_t num_attention_heads     = 32;
    std::size_t num_key_value_heads     = 8;
    std::size_t head_dim_               = 128;
    float       rms_norm_eps            = 1e-5f;
    float       rope_theta              = 10000.0f;
    int         sliding_window          = -1;
    int32_t     bos_token_id            = 1;
    int32_t     eos_token_id            = 2;
    backend::ops::RopeScaling rope_scaling;
    
    std::size_t head_dim() const { return head_dim_; }
};

struct MistralLayer {
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

class MistralModel {
public:
    // Takes model_dir to do custom JSON parsing for multimodal nested configs
    static MistralModel load(const parser::WeightMap& weights,
                             const std::string& model_dir,
                             const backend::Device& device);

    backend::Tensor forward(const std::vector<int32_t>& input_ids, KVCache& cache);

    KVCache make_cache(std::size_t max_seq_len) const;

    std::size_t vocab_size()   const noexcept { return cfg_.vocab_size; }
    int32_t     eos_token_id() const noexcept { return cfg_.eos_token_id; }
    int32_t     bos_token_id() const noexcept { return cfg_.bos_token_id; }

private:
    backend::Tensor layer_forward(const backend::Tensor& hidden,
                                  MistralLayer& layer,
                                  KVCache::LayerCache& lc,
                                  int seq_offset);

    MistralConfig             cfg_;
    backend::Device           device_ = backend::Device::cpu();
    backend::Tensor           embed_tokens_;
    backend::Tensor           norm_;
    backend::Tensor           lm_head_;
    std::vector<MistralLayer> layers_;
};

} // namespace tensor::models::mistral