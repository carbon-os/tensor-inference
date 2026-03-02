#include <transformers/inference/sampling/top_p.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace transformers::inference::sampling {

// Shared helper: copy last token logits to host as float32
static std::vector<float> last_logits_to_host(const backend::Tensor& logits) {
    const std::size_t vocab       = logits.shape().back();
    const std::size_t last_offset = logits.numel() - vocab;
    const std::size_t elem_bytes  = core::dtype_size(logits.dtype());

    std::vector<float> host(vocab);

    if (logits.dtype() == backend::DType::F32) {
        if (logits.device().is_cuda()) {
            cudaMemcpy(host.data(),
                       static_cast<const uint8_t*>(logits.data_ptr()) + last_offset * elem_bytes,
                       vocab * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            const float* src = logits.data_as<float>() + last_offset;
            std::copy(src, src + vocab, host.begin());
        }
    } else if (logits.dtype() == backend::DType::BF16) {
        // Copy raw bytes then convert
        std::vector<uint16_t> raw(vocab);
        if (logits.device().is_cuda()) {
            cudaMemcpy(raw.data(),
                       static_cast<const uint8_t*>(logits.data_ptr()) + last_offset * elem_bytes,
                       vocab * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        }
        for (std::size_t i = 0; i < vocab; ++i) {
            // BF16 → F32: shift mantissa left 16 bits
            uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
            std::memcpy(&host[i], &bits, sizeof(float));
        }
    } else {
        throw std::invalid_argument("sampling: unsupported logits dtype");
    }

    return host;
}

// ─────────────────────────────────────────────────────────────
//  TopP::sample
// ─────────────────────────────────────────────────────────────

int32_t TopP::sample(const backend::Tensor& logits) const {
    auto host = last_logits_to_host(logits);
    const std::size_t vocab = host.size();

    // Apply temperature
    if (temperature != 1.0f && temperature > 0.0f) {
        for (auto& v : host) v /= temperature;
    }

    // Softmax
    const float max_v = *std::max_element(host.begin(), host.end());
    float sum = 0.0f;
    for (auto& v : host) { v = std::exp(v - max_v); sum += v; }
    for (auto& v : host) v /= sum;

    // Sort indices by probability descending
    std::vector<int32_t> indices(vocab);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int32_t a, int32_t b) { return host[a] > host[b]; });

    // Find nucleus: cumulative prob >= p
    float cumsum = 0.0f;
    std::size_t cutoff = vocab;
    for (std::size_t i = 0; i < vocab; ++i) {
        cumsum += host[indices[i]];
        if (cumsum >= p) { cutoff = i + 1; break; }
    }

    // FIXED: Use static thread_local so the RNG state advances properly
    thread_local static std::mt19937_64 rng(seed.has_value() ? *seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float r = dist(rng) * cumsum;
    float acc = 0.0f;
    for (std::size_t i = 0; i < cutoff; ++i) {
        acc += host[indices[i]];
        if (r <= acc) return indices[i];
    }
    return indices[0];
}

// ─────────────────────────────────────────────────────────────
//  TopK::sample
// ─────────────────────────────────────────────────────────────

int32_t TopK::sample(const backend::Tensor& logits) const {
    auto host = last_logits_to_host(logits);
    const std::size_t vocab = host.size();
    const std::size_t k_eff = std::min(static_cast<std::size_t>(k), vocab);

    if (temperature != 1.0f && temperature > 0.0f) {
        for (auto& v : host) v /= temperature;
    }

    // Partial sort: get top-k indices
    std::vector<int32_t> indices(vocab);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k_eff, indices.end(),
                      [&](int32_t a, int32_t b) { return host[a] > host[b]; });
    indices.resize(k_eff);

    // Softmax over top-k
    float max_v = host[indices[0]];
    float sum = 0.0f;
    std::vector<float> probs(k_eff);
    for (std::size_t i = 0; i < k_eff; ++i) {
        probs[i] = std::exp(host[indices[i]] - max_v);
        sum += probs[i];
    }

    std::mt19937_64 rng(seed.has_value() ? *seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    float acc = 0.0f;
    for (std::size_t i = 0; i < k_eff; ++i) {
        acc += probs[i];
        if (r <= acc) return indices[i];
    }
    return indices[0];
}

} // namespace transformers::inference::sampling