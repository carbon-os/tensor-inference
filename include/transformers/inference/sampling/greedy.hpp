#pragma once

#include <transformers/backend/tensor.hpp>

#include <cstdint>

namespace transformers::inference::sampling {

// Always picks the highest-logit token. Deterministic.
struct Greedy {
    int32_t sample(const backend::Tensor& logits) const;
};

} // namespace transformers::inference::sampling