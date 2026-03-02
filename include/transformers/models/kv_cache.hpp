#pragma once

#include <transformers/backend/tensor.hpp>
#include <transformers/backend/device.hpp>

#include <cstddef>
#include <vector>

namespace transformers::models {

using backend::Device;
using backend::DType;
using backend::Shape;
using backend::Tensor;

// ─────────────────────────────────────────────────────────────
//  KVCache — per-layer key/value store
//
//  Sized from the model architecture at creation.
//  The model writes K and V into it each forward step.
//  Inference drives clearing between requests via clear().
// ─────────────────────────────────────────────────────────────

class KVCache {
public:
    struct LayerCache {
        Tensor keys;    // [max_seq_len, n_kv_heads, head_dim]
        Tensor values;  // [max_seq_len, n_kv_heads, head_dim]
    };

    KVCache(std::size_t n_layers,
            std::size_t n_kv_heads,
            std::size_t head_dim,
            std::size_t max_seq_len,
            DType       dtype,
            const Device& device);

    LayerCache&       layer(std::size_t i)       { return layers_[i]; }
    const LayerCache& layer(std::size_t i) const { return layers_[i]; }

    std::size_t n_layers()    const noexcept { return layers_.size(); }
    std::size_t seq_len()     const noexcept { return seq_len_; }
    std::size_t max_seq_len() const noexcept { return max_seq_len_; }
    bool        full()        const noexcept { return seq_len_ >= max_seq_len_; }

    void advance(std::size_t n = 1) noexcept { seq_len_ += n; }
    void clear()                    noexcept { seq_len_ = 0; }

    // Move-only (owns device Tensors)
    KVCache(KVCache&&) noexcept            = default;
    KVCache& operator=(KVCache&&) noexcept = default;
    KVCache(const KVCache&)                = delete;
    KVCache& operator=(const KVCache&)     = delete;

private:
    std::vector<LayerCache> layers_;
    std::size_t             seq_len_     = 0;
    std::size_t             max_seq_len_ = 0;
};

} // namespace transformers::models