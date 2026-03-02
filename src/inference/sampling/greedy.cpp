#include <transformers/inference/sampling/greedy.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace transformers::inference::sampling {

int32_t Greedy::sample(const backend::Tensor& logits) const {
    // Bring the last token's logits to host.
    // logits shape: [seq_len, vocab_size] — we want the last row.
    const std::size_t vocab = logits.shape().back();
    const std::size_t last_offset = (logits.numel() - vocab);

    // Copy just the last row to host.
    std::vector<float> host(vocab);
    if (logits.device().is_cuda()) {
        const std::size_t elem_bytes = core::dtype_size(logits.dtype());
        cudaMemcpy(
            host.data(),
            static_cast<const uint8_t*>(logits.data_ptr()) + last_offset * elem_bytes,
            vocab * elem_bytes,
            cudaMemcpyDeviceToHost
        );
        // If BF16, need to convert — for now assume F32 logits from lm_head
    } else {
        const float* src = logits.data_as<float>() + last_offset;
        std::copy(src, src + vocab, host.begin());
    }

    return static_cast<int32_t>(
        std::max_element(host.begin(), host.end()) - host.begin()
    );
}

} // namespace transformers::inference::sampling