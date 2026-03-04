#pragma once

#include <tensor/backend/tensor.hpp>

#include <cstdint>

namespace tensor::inference::sampling {

// Always picks the highest-logit token. Deterministic.
struct Greedy {
    int32_t sample(const backend::Tensor& logits) const;
};

} // namespace tensor::inference::sampling