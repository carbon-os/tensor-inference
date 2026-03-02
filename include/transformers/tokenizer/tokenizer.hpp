#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace transformers::tokenizer {

// ─────────────────────────────────────────────────────────────
//  Tokenizer — byte-level BPE, reads tokenizer.json
//
//  Covers: LLaMA 2/3, Mistral, Qwen, Gemma, Phi-3, and any
//  HuggingFace model that ships a BPE tokenizer.json.
// ─────────────────────────────────────────────────────────────

class Tokenizer {
public:
    static Tokenizer from_files(const std::string& model_dir);
    static Tokenizer from_file (const std::string& tokenizer_json);

    // text → token ids
    std::vector<int32_t> encode(const std::string& text,
                                bool add_special_tokens = true) const;

    // token ids → text
    std::string decode(const std::vector<int32_t>& ids,
                       bool skip_special_tokens = true) const;

    // Chat template (Jinja2 — basic implementation covering common patterns)
    struct Message { std::string role, content; };
    std::string apply_chat_template(const std::vector<Message>& messages,
                                    bool add_generation_prompt = true) const;

    std::size_t vocab_size()           const noexcept;
    int32_t     bos_token_id()         const noexcept;
    int32_t     eos_token_id()         const noexcept;
    int32_t     pad_token_id()         const noexcept;
    std::string id_to_token(int32_t)   const;
    int32_t     token_to_id(const std::string&) const; // -1 if absent

    Tokenizer(Tokenizer&&) noexcept            = default;
    Tokenizer& operator=(Tokenizer&&) noexcept = default;
    Tokenizer(const Tokenizer&)                = delete;
    Tokenizer& operator=(const Tokenizer&)     = delete;
    ~Tokenizer();

private:
    struct Impl;
    explicit Tokenizer(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

} // namespace transformers::tokenizer