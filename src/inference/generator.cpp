#include <tensor/inference/generator.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tensor::inference {

Generator::Generator(std::unique_ptr<ModelIface> iface,
                     const tokenizer::Tokenizer& tok,
                     std::size_t max_seq_len)
    : model_(std::move(iface))
    , tokenizer_(tok)
    , cache_(model_->make_cache(max_seq_len))
{}

int32_t Generator::sample_token(const backend::Tensor& logits,
                                 const Sampler& s) const {
    return std::visit([&](const auto& sampler) -> int32_t {
        return sampler.sample(logits);
    }, s);
}

GenerateResult Generator::generate(const std::vector<int32_t>& input_ids,
                                    const GenerateOptions& options) {
    if (input_ids.empty()) {
        throw std::invalid_argument("Generator::generate: empty input_ids");
    }

    GenerateResult result;
    result.stop_reason = "max_tokens";

    // Prefill — forward the full prompt
    backend::Tensor logits = model_->forward(input_ids, cache_);

    std::vector<int32_t> generated;
    generated.reserve(options.max_new_tokens);

    for (std::size_t step = 0; step < options.max_new_tokens; ++step) {

        int32_t next_token = sample_token(logits, options.sampler);

        if (next_token == model_->eos_token_id()) {
            result.stop_reason = "eos";
            break;
        }

        generated.push_back(next_token);
        result.tokens_generated++;

        if (options.on_token) {
            if (!options.on_token(next_token)) {
                result.stop_reason = "callback";
                break;
            }
        }

        if (!options.stop_strings.empty()) {
            const std::string decoded = tokenizer_.decode(generated, true);
            for (const auto& stop : options.stop_strings) {
                if (decoded.find(stop) != std::string::npos) {
                    result.stop_reason = "stop_string";
                    goto done;
                }
            }
        }

        // Single token decode step
        logits = model_->forward({next_token}, cache_);
    }

done:
    result.output_ids = std::move(generated);
    return result;
}

} // namespace tensor::inference