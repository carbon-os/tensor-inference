#include <tensor/models/llama/llama_config.hpp>

namespace tensor::models::llama {

LlamaConfig LlamaConfig::from_model_config(const parser::ModelConfig& cfg) {
    LlamaConfig c;
    c.vocab_size              = cfg.vocab_size();
    c.hidden_size             = cfg.hidden_size();
    c.intermediate_size       = cfg.intermediate_size();
    c.num_hidden_layers       = cfg.num_hidden_layers();
    c.num_attention_heads     = cfg.num_attention_heads();
    c.num_key_value_heads     = cfg.num_key_value_heads();
    c.max_position_embeddings = cfg.max_position_embeddings();
    c.rms_norm_eps            = cfg.rms_norm_eps();
    c.rope_theta              = cfg.rope_theta();
    c.bos_token_id            = cfg.bos_token_id();
    c.eos_token_id            = cfg.eos_token_id();
    c.tie_word_embeddings     = cfg.get_bool("tie_word_embeddings", false);

    // Parse Llama-3 / YaRN rope_scaling block.
    // rope_scaling_type() returns "" when absent.
    const std::string rs_type = cfg.rope_scaling_type();
    if (rs_type == "llama3" || rs_type == "yarn") {
        c.rope_scaling.enabled          = true;
        c.rope_scaling.factor           = cfg.rope_scaling_factor();
        c.rope_scaling.low_freq_factor  = cfg.rope_scaling_low_freq_factor();
        c.rope_scaling.high_freq_factor = cfg.rope_scaling_high_freq_factor();
        c.rope_scaling.original_max_pos =
            cfg.rope_scaling_original_max_position_embeddings();
    }

    return c;
}

} // namespace tensor::models::llama