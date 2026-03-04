#pragma once

#include <tensor/core/dtype.hpp>
#include <tensor/core/shape.hpp>
#include <tensor/core/tensor_view.hpp>
#include <tensor/backend/device.hpp>

#include <cstddef>
#include <vector>

namespace tensor::backend {

using core::DType;
using core::Shape;
using core::TensorView;

// ─────────────────────────────────────────────────────────────
//  Tensor — device-resident buffer. Move-only. Owns its memory.
// ─────────────────────────────────────────────────────────────

class Tensor {
public:
    Tensor() = default;
    ~Tensor();

    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // ── factory ──────────────────────────────────────────────

    // Allocate uninitialized device memory.
    static Tensor empty(const Shape& shape, DType dtype, const Device& device);

    // Allocate and zero-initialize.
    static Tensor zeros(const Shape& shape, DType dtype, const Device& device);

    // Upload a host TensorView to device memory (synchronous).
    static Tensor from_view(const TensorView& view, const Device& device);

    // ── metadata ─────────────────────────────────────────────

    DType         dtype()  const noexcept { return dtype_; }
    const Shape&  shape()  const noexcept { return shape_; }
    std::size_t   rank()   const noexcept { return shape_.rank(); }
    std::size_t   numel()  const noexcept { return shape_.numel(); }
    std::size_t   nbytes() const noexcept;
    const Device& device() const noexcept { return device_; }
    bool          valid()  const noexcept { return ptr_ != nullptr; }

    // Raw pointer — passed to CUDA kernels.
    void*       data_ptr()       noexcept { return ptr_; }
    const void* data_ptr() const noexcept { return ptr_; }

    template<typename T>       T* data_as()       noexcept { return static_cast<T*>(ptr_); }
    template<typename T> const T* data_as() const noexcept { return static_cast<const T*>(ptr_); }

    // ── host readback  (testing / debug) ─────────────────────

    std::vector<float>   to_host_f32() const;
    std::vector<int32_t> to_host_i32() const;

private:
    Tensor(void* ptr, DType dtype, Shape shape, Device device)
        : ptr_(ptr), dtype_(dtype), shape_(std::move(shape)), device_(std::move(device)) {}

    void*   ptr_   = nullptr;
    DType   dtype_ = DType::F32;
    Shape   shape_ = {};
    Device  device_;
};

} // namespace tensor::backend