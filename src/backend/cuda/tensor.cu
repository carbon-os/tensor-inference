#include <transformers/backend/tensor.hpp>
#include <transformers/core/dtype.hpp>

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>

namespace transformers::backend {

// ── helpers ──────────────────────────────────────────────────

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err)
        );
    }
}

// ── destructor / move ────────────────────────────────────────

Tensor::~Tensor() {
    if (ptr_ != nullptr) {
        if (device_.is_cuda()) {
            cudaSetDevice(device_.index());
            cudaFree(ptr_);
        } else {
            std::free(ptr_);
        }
        ptr_ = nullptr;
    }
}

Tensor::Tensor(Tensor&& o) noexcept
    : ptr_(o.ptr_), dtype_(o.dtype_), shape_(std::move(o.shape_)), device_(o.device_) {
    o.ptr_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& o) noexcept {
    if (this != &o) {
        // Release only the raw device/host buffer. Do NOT call this->~Tensor()
        // here: that would destroy shape_ (std::vector) and device_ (std::string),
        // and the subsequent move-assignments below would then operate on already-
        // destroyed objects, causing the tcache double-free detected by glibc.
        if (ptr_ != nullptr) {
            if (device_.is_cuda()) {
                cudaSetDevice(device_.index());
                cudaFree(ptr_);
            } else {
                std::free(ptr_);
            }
            ptr_ = nullptr;
        }
        ptr_    = o.ptr_;
        dtype_  = o.dtype_;
        shape_  = std::move(o.shape_);   // safe: shape_ is still a live object
        device_ = o.device_;             // safe: device_ is still a live object
        o.ptr_  = nullptr;
    }
    return *this;
}

// ── nbytes ───────────────────────────────────────────────────

std::size_t Tensor::nbytes() const noexcept {
    return numel() * core::dtype_size(dtype_);
}

// ── factory ──────────────────────────────────────────────────

Tensor Tensor::empty(const Shape& shape, DType dtype, const Device& device) {
    const std::size_t bytes = shape.numel() * core::dtype_size(dtype);
    void* ptr = nullptr;

    if (device.is_cuda()) {
        device.make_current();
        cuda_check(cudaMalloc(&ptr, bytes), "Tensor::empty cudaMalloc");
    } else {
        ptr = std::malloc(bytes);
        if (!ptr) throw std::bad_alloc();
    }

    return Tensor(ptr, dtype, shape, device);
}

Tensor Tensor::zeros(const Shape& shape, DType dtype, const Device& device) {
    const std::size_t bytes = shape.numel() * core::dtype_size(dtype);
    void* ptr = nullptr;

    if (device.is_cuda()) {
        device.make_current();
        cuda_check(cudaMalloc(&ptr, bytes), "Tensor::zeros cudaMalloc");
        cuda_check(cudaMemset(ptr, 0, bytes), "Tensor::zeros cudaMemset");
    } else {
        ptr = std::calloc(shape.numel(), core::dtype_size(dtype));
        if (!ptr) throw std::bad_alloc();
    }

    return Tensor(ptr, dtype, shape, device);
}

Tensor Tensor::from_view(const core::TensorView& view, const Device& device) {
    const std::size_t bytes = view.nbytes();
    void* ptr = nullptr;

    if (device.is_cuda()) {
        device.make_current();
        cuda_check(cudaMalloc(&ptr, bytes), "Tensor::from_view cudaMalloc");
        cuda_check(
            cudaMemcpy(ptr, view.data, bytes, cudaMemcpyHostToDevice),
            "Tensor::from_view cudaMemcpy H2D"
        );
    } else {
        ptr = std::malloc(bytes);
        if (!ptr) throw std::bad_alloc();
        std::memcpy(ptr, view.data, bytes);
    }

    return Tensor(ptr, view.dtype, view.shape, device);
}

// ── host readback ────────────────────────────────────────────

std::vector<float> Tensor::to_host_f32() const {
    if (dtype_ != DType::F32) {
        throw std::invalid_argument("Tensor::to_host_f32: dtype is not F32");
    }
    std::vector<float> out(numel());
    if (device_.is_cuda()) {
        device_.make_current();
        cuda_check(
            cudaMemcpy(out.data(), ptr_, nbytes(), cudaMemcpyDeviceToHost),
            "Tensor::to_host_f32 cudaMemcpy D2H"
        );
    } else {
        std::memcpy(out.data(), ptr_, nbytes());
    }
    return out;
}

std::vector<int32_t> Tensor::to_host_i32() const {
    if (dtype_ != DType::I32) {
        throw std::invalid_argument("Tensor::to_host_i32: dtype is not I32");
    }
    std::vector<int32_t> out(numel());
    if (device_.is_cuda()) {
        device_.make_current();
        cuda_check(
            cudaMemcpy(out.data(), ptr_, nbytes(), cudaMemcpyDeviceToHost),
            "Tensor::to_host_i32 cudaMemcpy D2H"
        );
    } else {
        std::memcpy(out.data(), ptr_, nbytes());
    }
    return out;
}

} // namespace transformers::backend