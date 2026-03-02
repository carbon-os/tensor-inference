#pragma once

#include <transformers/backend/tensor.hpp>

#include <cstdint>
#include <vector>

// ─────────────────────────────────────────────────────────────
//  backend::ops — CUDA compute primitives
//
//  Called exclusively by model implementations (llama_model.cpp etc.).
//  User code never includes this header.
//
//  All operations are synchronous with respect to the default CUDA
//  stream on the tensor's device. Each op returns a new Tensor;
//  inputs are never modified.
// ─────────────────────────────────────────────────────────────

namespace transformers::backend::ops {

// ── embedding ────────────────────────────────────────────────

// Token embedding lookup.
// weight: [vocab_size, hidden_size]
// indices: [seq_len]  (host vector)
// → [seq_len, hidden_size]
Tensor embedding(const Tensor& weight, const std::vector<int32_t>& indices);

// ── linear algebra ──────────────────────────────────────────

// C = A @ B^T  (the standard weight projection: x @ w^T)
// A: [M, K],  B: [N, K]  → [M, N]
// Supports F32 and BF16. Uses cuBLAS.
Tensor matmul_t(const Tensor& a, const Tensor& b);

// ── normalization ────────────────────────────────────────────

// RMS layer norm: out = (x / rms(x)) * weight
// x: [seq_len, hidden],  weight: [hidden]  → [seq_len, hidden]
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-5f);

// ── positional encoding ──────────────────────────────────────

// Per-dimension RoPE frequency scaling (Llama-3 NTK / YaRN).
// Default-constructed (apply_scaling = false) gives plain RoPE.
struct RopeScaling {
    bool  apply_scaling    = false;
    float factor           = 1.0f;
    float low_freq_factor  = 1.0f;
    float high_freq_factor = 1.0f;
    int   original_max_pos = 8192;
};

// Rotary Position Embedding with optional Llama-3 frequency scaling.
// x:          [seq_len, n_heads, head_dim]
// seq_offset: position of first token (0 for prefill, cache.seq_len() for decode)
// theta:      base frequency (rope_theta from config.json)
// scaling:    frequency scaling config (no-op when apply_scaling == false)
Tensor rope(const Tensor& x, int seq_offset, float theta, const RopeScaling& scaling);

// Backward-compatible 3-arg form — plain RoPE, no scaling.
Tensor rope(const Tensor& x, int seq_offset, float theta = 500000.0f);

// ── activations & elementwise ────────────────────────────────

// SiLU: out = x * sigmoid(x)
Tensor silu(const Tensor& x);

// Elementwise multiply: out = a * b  (same shape)
Tensor mul(const Tensor& a, const Tensor& b);

// Elementwise add: out = a + b  (same shape)
Tensor add(const Tensor& a, const Tensor& b);

// ── attention ────────────────────────────────────────────────

// Softmax along the last dimension.
Tensor softmax(const Tensor& x);

// Grouped-Query Attention (subsumes MHA when n_heads == n_kv_heads).
// q: [seq_len, n_heads,    head_dim]
// k: [kv_len,  n_kv_heads, head_dim]
// v: [kv_len,  n_kv_heads, head_dim]
// causal: apply causal mask (true during prefill, false for single decode token)
// sliding_window: max lookback distance (-1 for infinite)
// → [seq_len, n_heads, head_dim]
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal = true, int sliding_window = -1);

// ── shape / memory ───────────────────────────────────────────

// Reshape without copy. New shape must have same numel().
Tensor reshape(const Tensor& x, const Shape& shape);

// Transpose two dimensions. Returns a contiguous copy.
Tensor transpose(const Tensor& x, std::size_t dim_a, std::size_t dim_b);

// Cast to a different dtype.
Tensor cast(const Tensor& x, DType target);

// Concatenate along dim. All tensors must agree on all other dims.
Tensor cat(const std::vector<const Tensor*>& tensors, std::size_t dim);

// Slice [begin, end) along dim. Returns a contiguous copy.
Tensor slice(const Tensor& x, std::size_t dim, std::size_t begin, std::size_t end);

Tensor add_bias(const Tensor& x, const Tensor& bias);

// ── Gemma-specific ops ───────────────────────────────────────

// GELU activation: out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
Tensor gelu(const Tensor& x);

// RMS layer norm with an implicit +1.0f offset to the weights (used by Gemma)
Tensor rms_norm_offset(const Tensor& x, const Tensor& weight, float eps = 1e-6f);

// Elementwise scalar multiplication: out = x * scalar
Tensor mul(const Tensor& x, float scalar);

} // namespace transformers::backend::ops