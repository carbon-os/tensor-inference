#pragma once

#include <transformers/parser/config.hpp>

#include <cstddef>
#include <cstdint>

namespace transformers::models::llama {

// Per-dimension RoPE frequency scaling (Llama-3 NTK / YaRN).
// Mirrors the rope_scaling block in config.json.
struct RopeScalingConfig {
    bool  enabled            = false;
    float factor             = 1.0f;   // global divisor applied to low-freq dims
    float low_freq_factor    = 1.0f;   // wavelength > orig/low  → scale down by factor
    float high_freq_factor   = 4.0f;   // wavelength < orig/high → no scaling
    int   original_max_pos   = 8192;   // pre-scaling context window length
};

struct LlamaConfig {
    std::size_t vocab_size              = 32000;
    std::size_t hidden_size             = 4096;
    std::size_t intermediate_size       = 11008;
    std::size_t num_hidden_layers       = 32;
    std::size_t num_attention_heads     = 32;
    std::size_t num_key_value_heads     = 32;
    std::size_t max_position_embeddings = 4096;
    float       rms_norm_eps            = 1e-5f;
    float       rope_theta              = 500000.0f;
    int32_t     bos_token_id            = 128000;
    int32_t     eos_token_id            = 128001;

    // When true, lm_head shares the embed_tokens weight (no separate key in
    // the safetensors file).  Must be handled in LlamaModel::load.
    bool        tie_word_embeddings     = false;

    RopeScalingConfig rope_scaling;

    std::size_t head_dim() const noexcept {
        return hidden_size / num_attention_heads;
    }

    static LlamaConfig from_model_config(const parser::ModelConfig& cfg);
};

} // namespace transformers::models::llama