#pragma once

#include <cstddef>
#include <string>

namespace tensor::backend {

enum class DeviceType { CPU, CUDA };

class Device {
public:
    // Default constructor produces an inert CPU placeholder.
    // Required so Tensor (which holds a Device by value) can be
    // default-constructed for use in containers and LayerCache.
    Device() = default;

    static Device cpu();
    static Device cuda(int index = 0);

    DeviceType         type()            const noexcept { return type_; }
    int                index()           const noexcept { return index_; }
    const std::string& name()            const noexcept { return name_; }
    std::size_t        memory_capacity() const noexcept { return memory_bytes_; }

    bool is_cuda() const noexcept { return type_ == DeviceType::CUDA; }
    bool is_cpu()  const noexcept { return type_ == DeviceType::CPU;  }

    bool operator==(const Device& o) const noexcept {
        return type_ == o.type_ && index_ == o.index_;
    }

    void make_current() const;

private:
    Device(DeviceType type, int index, std::string name, std::size_t mem)
        : type_(type), index_(index), name_(std::move(name)), memory_bytes_(mem) {}

    DeviceType  type_         = DeviceType::CPU;
    int         index_        = 0;
    std::string name_;
    std::size_t memory_bytes_ = 0;
};

} // namespace tensor::backend