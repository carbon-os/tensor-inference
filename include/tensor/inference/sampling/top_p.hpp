#pragma once

#include <tensor/backend/tensor.hpp>

#include <cstdint>
#include <optional>

namespace tensor::inference::sampling {

// Nucleus sampling: sample from the smallest token set whose
// cumulative probability >= p. Temperature is applied first.
struct TopP {
    float                   temperature = 1.0f;
    float                   p           = 0.95f;
    std::optional<uint64_t> seed;

    int32_t sample(const backend::Tensor& logits) const;
};

// Top-K sampling: sample from the k highest-probability tokens.
struct TopK {
    float                   temperature = 1.0f;
    uint32_t                k           = 40;
    std::optional<uint64_t> seed;

    int32_t sample(const backend::Tensor& logits) const;
};

} // namespace tensor::inference::sampling