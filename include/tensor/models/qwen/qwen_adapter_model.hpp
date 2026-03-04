#pragma once

#include <tensor/adapter/adapter_weights.hpp>
#include <tensor/models/qwen/qwen_model.hpp>

#include <vector>

namespace tensor::models::qwen {

// ─────────────────────────────────────────────────────────────
//  QwenAdapterModel
//
//  Extends QwenModel with a loaded tensor adapter. The base
//  weights stay frozen. LoRA deltas are applied per-projection
//  at every layer inside the adapter's target range.
//
//  Usage:
//    auto base    = QwenModel::load(weights, config, device);
//    auto adapter = AdapterLoader::load(adapter_dir, "qwen2",
//                                       config.num_hidden_layers(),
//                                       device);
//    auto model   = QwenAdapterModel(std::move(base),
//                                    std::move(adapter));
//    auto gen     = Generator::create(model, tokenizer, max_seq);
// ─────────────────────────────────────────────────────────────

class QwenAdapterModel : public QwenModel {
public:
    QwenAdapterModel(QwenModel&&                     base,
                     tensor::adapter::AdapterWeights adapter);

    // Full forward pass with LoRA deltas injected where present
    backend::Tensor forward(const std::vector<int32_t>& input_ids,
                            KVCache&                     cache);

    // Inherited from QwenModel — no need to redeclare:
    //   make_cache(), vocab_size(), eos_token_id()

private:
    tensor::adapter::AdapterWeights adapter_;

    backend::Tensor layer_forward_adapted(
        const backend::Tensor&               hidden,
        QwenLayer&                           layer,
        KVCache::LayerCache&                 lc,
        int                                  seq_offset,
        const tensor::adapter::AdapterLayer& al);
};

} // namespace tensor::models::qwen