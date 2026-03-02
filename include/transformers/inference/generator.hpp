#pragma once

#include <transformers/backend/tensor.hpp>
#include <transformers/models/kv_cache.hpp>
#include <transformers/tokenizer/tokenizer.hpp>
#include <transformers/inference/sampling/greedy.hpp>
#include <transformers/inference/sampling/top_p.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace transformers::inference {

// ─────────────────────────────────────────────────────────────
//  Sampler variant
// ─────────────────────────────────────────────────────────────

using Sampler = std::variant<
    sampling::Greedy,
    sampling::TopP,
    sampling::TopK
>;

// ─────────────────────────────────────────────────────────────
//  GenerateOptions
// ─────────────────────────────────────────────────────────────

struct GenerateOptions {
    std::size_t  max_new_tokens = 256;
    Sampler      sampler        = sampling::Greedy{};
    std::vector<std::string>              stop_strings;
    std::function<bool(int32_t token_id)> on_token;
};

// ─────────────────────────────────────────────────────────────
//  GenerateResult
// ─────────────────────────────────────────────────────────────

struct GenerateResult {
    std::vector<int32_t> output_ids;
    std::size_t          tokens_generated = 0;
    std::string          stop_reason;   // "eos" | "max_tokens" | "stop_string" | "callback"
};

// ─────────────────────────────────────────────────────────────
//  Generator
// ─────────────────────────────────────────────────────────────

class Generator {
public:
    template<typename M>
    static Generator create(M& model,
                            const tokenizer::Tokenizer& tok,
                            std::size_t max_seq_len = 4096) {
        struct Wrapper : ModelIface {
            M& m;
            explicit Wrapper(M& model) : m(model) {}
            backend::Tensor forward(const std::vector<int32_t>& ids,
                                    models::KVCache& cache) override {
                return m.forward(ids, cache);
            }
            std::size_t     vocab_size()           const override { return m.vocab_size(); }
            int32_t         eos_token_id()         const override { return m.eos_token_id(); }
            models::KVCache make_cache(std::size_t n) const override { return m.make_cache(n); }
        };
        return Generator(std::make_unique<Wrapper>(model), tok, max_seq_len);
    }

    GenerateResult generate(const std::vector<int32_t>& input_ids,
                            const GenerateOptions& options = {});

    void reset() { cache_.clear(); }

private:
    struct ModelIface {
        virtual backend::Tensor forward(const std::vector<int32_t>&,
                                        models::KVCache&)       = 0;
        virtual std::size_t     vocab_size()              const = 0;
        virtual int32_t         eos_token_id()            const = 0;
        virtual models::KVCache make_cache(std::size_t)   const = 0;
        virtual ~ModelIface() = default;
    };

    Generator(std::unique_ptr<ModelIface> iface,
              const tokenizer::Tokenizer& tok,
              std::size_t max_seq_len);

    std::unique_ptr<ModelIface>  model_;
    const tokenizer::Tokenizer&  tokenizer_;
    models::KVCache              cache_;

    int32_t sample_token(const backend::Tensor& logits, const Sampler& s) const;
};

} // namespace transformers::inference