#include <transformers/backend/device.hpp>
#include <transformers/parser/errors.hpp>

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace transformers::backend {

Device Device::cpu() {
    return Device(DeviceType::CPU, 0, "CPU", 0);
}

Device Device::cuda(int index) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        throw std::runtime_error("Device::cuda: no CUDA devices available");
    }
    if (index < 0 || index >= count) {
        throw std::runtime_error(
            "Device::cuda: index " + std::to_string(index) +
            " out of range (found " + std::to_string(count) + " devices)"
        );
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, index);

    return Device(
        DeviceType::CUDA,
        index,
        std::string(prop.name),
        static_cast<std::size_t>(prop.totalGlobalMem)
    );
}

void Device::make_current() const {
    if (type_ == DeviceType::CUDA) {
        cudaSetDevice(index_);
    }
}

} // namespace transformers::backend