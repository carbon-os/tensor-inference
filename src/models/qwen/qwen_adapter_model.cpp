#include <tensor/models/qwen/qwen_adapter_model.hpp>
#include <tensor/adapter/adapter_weights.hpp>
#include <tensor/backend/ops.hpp>

#include <cuda_runtime.h>
#include <optional>
#include <utility>

namespace ops = tensor::backend::ops;

namespace tensor::models::qwen {

// ─────────────────────────────────────────────────────────────
//  apply_lora
//
//  x      : [seq, in_features]
//  lora_A : [rank, in_features]   → matmul_t(x, A) = [seq, rank]
//  lora_B : [out_features, rank]  → matmul_t(·, B) = [seq, out]
//  scale  : alpha / rank
// ─────────────────────────────────────────────────────────────

static backend::Tensor apply_lora(
    const backend::Tensor&                 x,
    const tensor::adapter::LoraProjection& proj,
    float                                  scale)
{
    auto down = ops::matmul_t(x, proj.lora_A);    // [seq, rank]
    auto up   = ops::matmul_t(down, proj.lora_B); // [seq, out]
    return ops::mul(up, scale);
}

// Returns base_out unchanged when proj is empty, otherwise adds the delta.
// base_out taken by value — callers always pass a temporary so this is a
// move-in. No copy of Tensor ever attempted.
static backend::Tensor add_if_present(
    const backend::Tensor&                                 x,
    backend::Tensor                                        base_out,
    const std::optional<tensor::adapter::LoraProjection>&  proj,
    float                                                  scale)
{
    if (!proj.has_value()) return std::move(base_out);
    return ops::add(base_out, apply_lora(x, *proj, scale));
}

// ─────────────────────────────────────────────────────────────
//  Construction
//
//  Delegate to QwenModel's move constructor — it has direct
//  access to its own members so the protected restriction
//  does not apply. adapter_ is initialised alongside it.
// ─────────────────────────────────────────────────────────────

QwenAdapterModel::QwenAdapterModel(QwenModel&&                     base,
                                   tensor::adapter::AdapterWeights adapter)
    : QwenModel(std::move(base))
    , adapter_(std::move(adapter))
{}

// ─────────────────────────────────────────────────────────────
//  layer_forward_adapted
//
//  Mirrors QwenModel::layer_forward exactly, with LoRA deltas
//  added after the base matmul at each injected projection.
// ─────────────────────────────────────────────────────────────

backend::Tensor QwenAdapterModel::layer_forward_adapted(
    const backend::Tensor&               hidden,
    QwenLayer&                           layer,
    KVCache::LayerCache&                 lc,
    int                                  seq_offset,
    const tensor::adapter::AdapterLayer& al)
{
    const float scale = adapter_.scale();

    const int seq_len  = static_cast<int>(hidden.shape()[0]);
    const int n_heads  = static_cast<int>(cfg_.num_attention_heads);
    const int n_kv     = static_cast<int>(cfg_.num_key_value_heads);
    const int head_dim = static_cast<int>(cfg_.head_dim());
    const int hidden_s = static_cast<int>(cfg_.hidden_size);

    backend::Tensor normed = ops::rms_norm(hidden, layer.input_layernorm,
                                           cfg_.rms_norm_eps);

    // ── Attention projections (base + optional LoRA delta) ────

    backend::Tensor q = add_if_present(normed,
        ops::add_bias(ops::matmul_t(normed, layer.q_proj_weight), layer.q_proj_bias),
        al.q_proj, scale);

    backend::Tensor k = add_if_present(normed,
        ops::add_bias(ops::matmul_t(normed, layer.k_proj_weight), layer.k_proj_bias),
        al.k_proj, scale);

    backend::Tensor v = add_if_present(normed,
        ops::add_bias(ops::matmul_t(normed, layer.v_proj_weight), layer.v_proj_bias),
        al.v_proj, scale);

    // ── Reshape + RoPE ────────────────────────────────────────

    q = ops::reshape(q, {static_cast<std::size_t>(seq_len),
                         static_cast<std::size_t>(n_heads),
                         static_cast<std::size_t>(head_dim)});
    k = ops::reshape(k, {static_cast<std::size_t>(seq_len),
                         static_cast<std::size_t>(n_kv),
                         static_cast<std::size_t>(head_dim)});
    v = ops::reshape(v, {static_cast<std::size_t>(seq_len),
                         static_cast<std::size_t>(n_kv),
                         static_cast<std::size_t>(head_dim)});

    q = ops::rope(q, seq_offset, cfg_.rope_theta);
    k = ops::rope(k, seq_offset, cfg_.rope_theta);

    {
        const std::size_t kv_row_bytes =
            n_kv * head_dim * core::dtype_size(k.dtype());
        cudaMemcpy(
            static_cast<uint8_t*>(lc.keys.data_ptr())   + seq_offset * kv_row_bytes,
            k.data_ptr(), k.nbytes(), cudaMemcpyDeviceToDevice);
        cudaMemcpy(
            static_cast<uint8_t*>(lc.values.data_ptr()) + seq_offset * kv_row_bytes,
            v.data_ptr(), v.nbytes(), cudaMemcpyDeviceToDevice);
    }

    const std::size_t kv_len =
        static_cast<std::size_t>(seq_offset) + static_cast<std::size_t>(seq_len);
    backend::Tensor k_ctx = ops::slice(lc.keys,   0, 0, kv_len);
    backend::Tensor v_ctx = ops::slice(lc.values, 0, 0, kv_len);

    backend::Tensor attn_out =
        ops::attention(q, k_ctx, v_ctx, seq_len > 1 /* causal */);
    attn_out = ops::reshape(attn_out,
        {static_cast<std::size_t>(seq_len),
         static_cast<std::size_t>(hidden_s)});

    // ── Output projection (base + optional LoRA delta) ────────

    backend::Tensor attn_proj = add_if_present(attn_out,
        ops::matmul_t(attn_out, layer.o_proj_weight),
        al.o_proj, scale);

    backend::Tensor residual1 = ops::add(hidden, attn_proj);

    // ── FFN (base + optional LoRA delta on gate / up / down) ──

    backend::Tensor normed2 = ops::rms_norm(residual1,
                                            layer.post_attention_layernorm,
                                            cfg_.rms_norm_eps);

    backend::Tensor gate = add_if_present(normed2,
        ops::matmul_t(normed2, layer.gate_proj_weight),
        al.gate_proj, scale);

    backend::Tensor up = add_if_present(normed2,
        ops::matmul_t(normed2, layer.up_proj_weight),
        al.up_proj, scale);

    backend::Tensor act = ops::mul(ops::silu(gate), up);

    backend::Tensor down = add_if_present(act,
        ops::matmul_t(act, layer.down_proj_weight),
        al.down_proj, scale);

    return ops::add(residual1, down);
}

// ─────────────────────────────────────────────────────────────
//  forward
// ─────────────────────────────────────────────────────────────

backend::Tensor QwenAdapterModel::forward(
    const std::vector<int32_t>& input_ids,
    KVCache&                    cache)
{
    device_.make_current();
    const int seq_offset = static_cast<int>(cache.seq_len());

    backend::Tensor hidden = ops::embedding(embed_tokens_, input_ids);

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        hidden = layer_forward_adapted(
            hidden,
            layers_[i],
            cache.layer(i),
            seq_offset,
            adapter_.layers[i]);  // empty AdapterLayer outside target range → all no-ops
    }

    hidden = ops::rms_norm(hidden, norm_, cfg_.rms_norm_eps);
    auto logits = ops::matmul_t(hidden, lm_head_);

    cache.advance(input_ids.size());
    return logits;
}

} // namespace tensor::models::qwen