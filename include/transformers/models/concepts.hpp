#pragma once

#include <transformers/backend/tensor.hpp>
#include <transformers/models/kv_cache.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace transformers::models {

using backend::Tensor;

// Any decoder model that Generator can drive.
template<typename M>
concept CausalLM = requires(M m,
                            const std::vector<int32_t>& tokens,
                            KVCache& cache) {
    { m.forward(tokens, cache) } -> std::same_as<Tensor>;  // logits [seq, vocab]
    { m.vocab_size()   }         -> std::same_as<std::size_t>;
    { m.eos_token_id() }         -> std::same_as<int32_t>;
};

} // namespace transformers::models