#include <tensor/models/kv_cache.hpp>

namespace tensor::models {

KVCache::KVCache(std::size_t n_layers,
                 std::size_t n_kv_heads,
                 std::size_t head_dim,
                 std::size_t max_seq_len,
                 DType       dtype,
                 const Device& device)
    : max_seq_len_(max_seq_len)
{
    layers_.reserve(n_layers);
    const Shape kv_shape = {max_seq_len, n_kv_heads, head_dim};
    for (std::size_t i = 0; i < n_layers; ++i) {
        LayerCache lc;
        lc.keys   = Tensor::zeros(kv_shape, dtype, device);
        lc.values = Tensor::zeros(kv_shape, dtype, device);
        layers_.push_back(std::move(lc));
    }
}

} // namespace tensor::models