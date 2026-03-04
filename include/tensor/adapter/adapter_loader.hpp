#pragma once

#include <tensor/adapter/adapter_weights.hpp>
#include <tensor/backend/device.hpp>

#include <cstddef>
#include <string>

namespace tensor::adapter {

// ─────────────────────────────────────────────────────────────
//  AdapterLoader
//
//  Reads an adapter directory produced by tensor-adapt:
//
//    my-adapter/
//    ├── adapter.safetensors
//    ├── adapter.json
//    └── tokenizer/          ← not used here
//
//  Validates:
//    • architecture field in adapter.json == expected_arch
//    • centroid_count > 0 (file was fully merged)
//    • every expected weight key is present in the safetensors
//
//  Throws std::runtime_error on any validation failure.
// ─────────────────────────────────────────────────────────────

class AdapterLoader {
public:
    // num_layers must match the loaded base model so the layers
    // vector is sized correctly.  Pass base.cfg_.num_hidden_layers.
    static AdapterWeights load(
        const std::string&     adapter_dir,
        const std::string&     expected_arch,   // "qwen2"
        std::size_t            num_layers,
        const backend::Device& device
    );
};

} // namespace tensor::adapter