#include <tensor/tokenizer/tokenizer.hpp>
#include <tensor/parser/errors.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;
using tensor::parser::ParseError;

namespace tensor::tokenizer {

// ─────────────────────────────────────────────────────────────
//  UTF-8 helpers
// ─────────────────────────────────────────────────────────────

static std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6)  & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// ─────────────────────────────────────────────────────────────
//  Byte ↔ Unicode mapping
// ─────────────────────────────────────────────────────────────

static std::array<std::string, 256> make_byte_encoder() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

    std::vector<uint32_t> cs(bs.begin(), bs.end());
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }

    std::array<std::string, 256> enc;
    for (std::size_t i = 0; i < bs.size(); ++i)
        enc[static_cast<uint8_t>(bs[i])] = codepoint_to_utf8(cs[i]);
    return enc;
}

static std::array<std::string, 256> BYTE_ENCODER = make_byte_encoder();

static std::unordered_map<std::string, uint8_t> make_byte_decoder() {
    std::unordered_map<std::string, uint8_t> dec;
    for (int b = 0; b < 256; ++b)
        dec[BYTE_ENCODER[b]] = static_cast<uint8_t>(b);
    return dec;
}

static std::unordered_map<std::string, uint8_t> BYTE_DECODER = make_byte_decoder();

// ─────────────────────────────────────────────────────────────
//  BPE encode of a single pre-tokenized word
// ─────────────────────────────────────────────────────────────

using MergeRanks = std::unordered_map<std::string, int>;

static std::vector<std::string> bpe_encode_word(
    const std::string& word_utf8,
    const MergeRanks&  merge_ranks)
{
    std::vector<std::string> tokens;
    std::size_t i = 0;
    while (i < word_utf8.size()) {
        unsigned char c = static_cast<unsigned char>(word_utf8[i]);
        std::size_t char_len = 1;
        if      (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;
        tokens.push_back(word_utf8.substr(i, char_len));
        i += char_len;
    }

    if (tokens.size() <= 1) return tokens;

    while (true) {
        int best_rank = std::numeric_limits<int>::max();
        int best_pos  = -1;
        for (int j = 0; j < static_cast<int>(tokens.size()) - 1; ++j) {
            auto it = merge_ranks.find(tokens[j] + " " + tokens[j + 1]);
            if (it != merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = j;
            }
        }
        if (best_pos == -1) break;

        tokens[best_pos] += tokens[best_pos + 1];
        tokens.erase(tokens.begin() + best_pos + 1);
    }

    return tokens;
}

// ─────────────────────────────────────────────────────────────
//  Tokenizer::Impl
// ─────────────────────────────────────────────────────────────

struct Tokenizer::Impl {
    std::unordered_map<std::string, int32_t> vocab;
    std::vector<std::string>                 id_to_tok;
    MergeRanks                               merge_ranks;

    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t pad_id = -1;
    std::unordered_map<int32_t, std::string> special_tokens;
    std::unordered_map<std::string, int32_t> special_vocab;

    bool        add_bos_token          = true;
    bool        add_eos_token          = false;
    std::string chat_template;
    std::string default_system_prompt  = "You are a helpful assistant.";

    std::regex pre_tokenize_regex{
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)",
        std::regex::optimize
    };

    void load_tokenizer_json(const std::string& path);
    void load_tokenizer_config(const std::string& path);
    void encode_piece(const std::string& piece, std::vector<int32_t>& out) const;
};

void Tokenizer::Impl::load_tokenizer_json(const std::string& path) {
    std::cerr << "[DEBUG tokenizer] Opening: " << path << "\n";
    std::ifstream f(path);
    if (!f.is_open()) throw ParseError("cannot open tokenizer.json: " + path);

    json root;
    try { root = json::parse(f); }
    catch (const json::exception& e) {
        throw ParseError(std::string("tokenizer.json parse error: ") + e.what());
    }

    // ── vocabulary ───────────────────────────────────────────
    const auto& model  = root["model"];
    const auto& jvocab = model["vocab"];

    id_to_tok.resize(jvocab.size());
    for (const auto& [token, id_val] : jvocab.items()) {
        int32_t id = id_val.get<int32_t>();
        vocab[token] = id;
        if (static_cast<std::size_t>(id) >= id_to_tok.size())
            id_to_tok.resize(static_cast<std::size_t>(id) + 1);
        id_to_tok[id] = token;
    }
    std::cerr << "[DEBUG tokenizer] Loaded vocab size: " << vocab.size() << "\n";

    // ── merges ───────────────────────────────────────────────
    int rank = 0, array_merges = 0, string_merges = 0;
    for (const auto& merge : model["merges"]) {
        if (merge.is_string()) {
            merge_ranks[merge.get<std::string>()] = rank++;
            ++string_merges;
        } else if (merge.is_array() && merge.size() >= 2 &&
                   merge[0].is_string() && merge[1].is_string()) {
            merge_ranks[merge[0].get<std::string>() + " " +
                        merge[1].get<std::string>()] = rank++;
            ++array_merges;
        } else {
            std::cerr << "[DEBUG tokenizer] Warning: unrecognised merge at rank "
                      << rank << "\n";
        }
    }
    std::cerr << "[DEBUG tokenizer] Loaded " << rank << " merges ("
              << string_merges << " strings, " << array_merges << " arrays).\n";

    // ── added / special tokens ───────────────────────────────
    if (root.contains("added_tokens")) {
        int count = 0;
        for (const auto& t : root["added_tokens"]) {
            if (!t.contains("content") || !t["content"].is_string()) {
                std::cerr << "[DEBUG tokenizer] Warning: skipped added_token "
                             "missing string 'content'\n";
                continue;
            }
            const std::string content = t["content"].get<std::string>();
            const int32_t    id      = t["id"].get<int32_t>();
            special_tokens[id]      = content;
            special_vocab[content]  = id;
            vocab[content]          = id;
            if (static_cast<std::size_t>(id) >= id_to_tok.size())
                id_to_tok.resize(static_cast<std::size_t>(id) + 1);
            id_to_tok[id] = content;
            ++count;
        }
        std::cerr << "[DEBUG tokenizer] Loaded " << count << " added_tokens.\n";
    }

    // ── resolve BOS / EOS from special-token name priority lists ─
    //
    // For Qwen2 chat / LoRA:
    //   BOS — <|endoftext|> (151643) — but add_bos_token=false so it's never prepended
    //   EOS — <|im_end|>    (151645) — ends every assistant turn in ChatML
    //
    // <|endoftext|> is kept as final fallback so base-model usage still works.
    static const std::vector<std::string> BOS_NAMES = {
        "<s>", "<|begin_of_text|>", "<|endoftext|>"
    };
    static const std::vector<std::string> EOS_NAMES = {
        "</s>", "<|end_of_text|>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>"
    };

    for (const auto& name : BOS_NAMES) {
        auto it = special_vocab.find(name);
        if (it != special_vocab.end()) { bos_id = it->second; break; }
    }
    for (const auto& name : EOS_NAMES) {
        auto it = special_vocab.find(name);
        if (it != special_vocab.end()) { eos_id = it->second; break; }
    }
}

void Tokenizer::Impl::load_tokenizer_config(const std::string& path) {
    std::cerr << "[DEBUG tokenizer] Opening config: " << path << "\n";
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[DEBUG tokenizer] tokenizer_config.json not found.\n";
        return;
    }

    json root;
    try { root = json::parse(f); }
    catch (...) {
        std::cerr << "[DEBUG tokenizer] Error parsing tokenizer_config.json\n";
        return;
    }

    // Resolve a token config key to its integer id.
    //
    // Priority (highest → lowest):
    //   1. String value  — "eos_token": "<|im_end|>"      → vocab lookup
    //   2. Object value  — "eos_token": {"content": "..."}→ vocab lookup
    //   3. Integer value — "eos_token": 151645             → direct
    //   4. *_id sibling  — "eos_token_id": 151643          → direct (fallback only)
    //   5. null / absent → -1 (no override)
    //
    // The *_id sibling is intentionally lowest priority.  For Qwen2, the config
    // has "eos_token": "<|im_end|>" AND "eos_token_id": 151643 (the base EOS).
    // We want the string value — <|im_end|>=151645 — to win so ChatML generation
    // stops at turn boundaries rather than at <|endoftext|>.
    auto get_token_id = [&](const std::string& key) -> int32_t {
        // Helper: look up a string in special_vocab then regular vocab.
        auto lookup_str = [&](const std::string& tok) -> int32_t {
            auto it = special_vocab.find(tok);
            if (it != special_vocab.end()) return it->second;
            auto it2 = vocab.find(tok);
            if (it2 != vocab.end()) return it2->second;
            std::cerr << "[DEBUG tokenizer] Warning: '" << key
                      << "' token '" << tok << "' not found in vocab.\n";
            return -1;
        };

        if (root.contains(key)) {
            const auto& v = root[key];

            if (v.is_null())
                return -1;   // explicit null → no override (e.g. Qwen2 bos_token)

            if (v.is_string())
                return lookup_str(v.get<std::string>());

            if (v.is_object() && v.contains("content") && v["content"].is_string())
                return lookup_str(v["content"].get<std::string>());

            if (v.is_number_integer())
                return v.get<int32_t>();

            if (v.is_array() && !v.empty() && v[0].is_number_integer()) {
                std::cerr << "[DEBUG tokenizer] Note: '" << key
                          << "' is an array, using first element.\n";
                return v[0].get<int32_t>();
            }

            std::cerr << "[DEBUG tokenizer] Warning: unrecognised format for '"
                      << key << "'\n";
        }

        // Fall back to the explicit *_id sibling only when the token key itself
        // is absent or produced no result above.
        const std::string id_key = key + "_id";
        if (root.contains(id_key) && root[id_key].is_number_integer())
            return root[id_key].get<int32_t>();

        return -1;
    };

    // Config values override the defaults resolved from added_tokens.
    // Null → -1 → no override, so bos_id keeps its value from load_tokenizer_json.
    if (int32_t id = get_token_id("bos_token"); id != -1) bos_id = id;
    if (int32_t id = get_token_id("eos_token"); id != -1) eos_id = id;
    if (int32_t id = get_token_id("pad_token"); id != -1) pad_id = id;

    std::cerr << "[DEBUG tokenizer] Resolved BOS: " << bos_id
              << ", EOS: " << eos_id
              << ", PAD: " << pad_id << "\n";

    if (root.contains("add_bos_token") && root["add_bos_token"].is_boolean())
        add_bos_token = root["add_bos_token"].get<bool>();
    if (root.contains("add_eos_token") && root["add_eos_token"].is_boolean())
        add_eos_token = root["add_eos_token"].get<bool>();

    if (root.contains("chat_template")) {
        const auto& ct = root["chat_template"];
        if (ct.is_string()) {
            chat_template = ct.get<std::string>();
        } else if (ct.is_array()) {
            std::cerr << "[DEBUG tokenizer] Note: 'chat_template' is an array, "
                         "skipping automatic extraction.\n";
        }
    }
}

void Tokenizer::Impl::encode_piece(const std::string& piece,
                                    std::vector<int32_t>& out) const {
    auto it = special_vocab.find(piece);
    if (it != special_vocab.end()) { out.push_back(it->second); return; }

    std::string byte_encoded;
    for (unsigned char c : piece)
        byte_encoded += BYTE_ENCODER[c];

    for (const auto& tok : bpe_encode_word(byte_encoded, merge_ranks)) {
        auto it2 = vocab.find(tok);
        if (it2 != vocab.end()) out.push_back(it2->second);
    }
}

// ─────────────────────────────────────────────────────────────
//  Tokenizer factories
// ─────────────────────────────────────────────────────────────

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::~Tokenizer() = default;

Tokenizer Tokenizer::from_file(const std::string& tokenizer_json) {
    auto impl = std::make_unique<Impl>();
    impl->load_tokenizer_json(tokenizer_json);
    return Tokenizer(std::move(impl));
}

Tokenizer Tokenizer::from_files(const std::string& model_dir) {
    auto impl = std::make_unique<Impl>();
    fs::path dir(model_dir);
    impl->load_tokenizer_json((dir / "tokenizer.json").string());
    impl->load_tokenizer_config((dir / "tokenizer_config.json").string());
    return Tokenizer(std::move(impl));
}

// ─────────────────────────────────────────────────────────────
//  encode
// ─────────────────────────────────────────────────────────────

std::vector<int32_t> Tokenizer::encode(const std::string& text,
                                        bool add_special_tokens) const {
    std::vector<int32_t> ids;

    if (add_special_tokens && impl_->add_bos_token && impl_->bos_id != -1)
        ids.push_back(impl_->bos_id);

    auto words_begin = std::sregex_iterator(text.begin(), text.end(),
                                             impl_->pre_tokenize_regex);
    for (auto it = words_begin; it != std::sregex_iterator(); ++it)
        impl_->encode_piece(it->str(), ids);

    if (add_special_tokens && impl_->add_eos_token && impl_->eos_id != -1)
        ids.push_back(impl_->eos_id);

    return ids;
}

// ─────────────────────────────────────────────────────────────
//  decode
// ─────────────────────────────────────────────────────────────

std::string Tokenizer::decode(const std::vector<int32_t>& ids,
                               bool skip_special_tokens) const {
    std::string byte_str;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= impl_->id_to_tok.size())
            continue;
        if (skip_special_tokens && impl_->special_tokens.count(id))
            continue;
        byte_str += impl_->id_to_tok[id];
    }

    std::string result;
    std::size_t i = 0;
    while (i < byte_str.size()) {
        unsigned char c = static_cast<unsigned char>(byte_str[i]);
        std::size_t char_len = 1;
        if      (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;
        std::string ch = byte_str.substr(i, char_len);
        auto it = BYTE_DECODER.find(ch);
        result += (it != BYTE_DECODER.end()) ? static_cast<char>(it->second) : ch[0];
        i += char_len;
    }

    // Replace SentencePiece space character (U+2581) with a standard space.
    const std::string sp_space = "\xe2\x96\x81";
    for (std::size_t pos = 0;
         (pos = result.find(sp_space, pos)) != std::string::npos; )
    {
        result.replace(pos, sp_space.length(), " ");
        pos += 1;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────
//  apply_chat_template
// ─────────────────────────────────────────────────────────────

std::string Tokenizer::apply_chat_template(const std::vector<Message>& messages,
                                            bool add_generation_prompt) const {
    const std::string& tmpl = impl_->chat_template;

    const bool is_llama3 = tmpl.find("start_header_id") != std::string::npos;
    const bool is_chatml = tmpl.find("im_start")        != std::string::npos;

    std::ostringstream out;

    if (is_llama3) {
        // ── LLaMA-3 / LLaMA-3.1 ─────────────────────────────
        out << "<|begin_of_text|>";
        for (const auto& msg : messages) {
            out << "<|start_header_id|>" << msg.role << "<|end_header_id|>\n\n"
                << msg.content << "<|eot_id|>";
        }
        if (add_generation_prompt)
            out << "<|start_header_id|>assistant<|end_header_id|>\n\n";

    } else if (is_chatml) {
        // ── ChatML — Qwen2 / Qwen2.5 ─────────────────────────
        //
        // Must match dataset-format's render_chatml() exactly:
        // inject default system turn when absent so the adapter's learned
        // activation pattern fires at inference as it did during training.
        bool has_system = !messages.empty() && messages[0].role == "system";
        if (!has_system) {
            out << "<|im_start|>system\n"
                << impl_->default_system_prompt
                << "<|im_end|>\n";
        }
        for (const auto& msg : messages) {
            out << "<|im_start|>" << msg.role << "\n"
                << msg.content   << "<|im_end|>\n";
        }
        if (add_generation_prompt)
            out << "<|im_start|>assistant\n";

    } else {
        // ── Generic fallback ─────────────────────────────────
        for (const auto& msg : messages)
            out << msg.role << ": " << msg.content << "\n";
        if (add_generation_prompt)
            out << "assistant: ";
    }

    return out.str();
}

// ─────────────────────────────────────────────────────────────
//  vocabulary accessors
// ─────────────────────────────────────────────────────────────

std::size_t Tokenizer::vocab_size()   const noexcept { return impl_->id_to_tok.size(); }
int32_t     Tokenizer::bos_token_id() const noexcept { return impl_->bos_id; }
int32_t     Tokenizer::eos_token_id() const noexcept { return impl_->eos_id; }
int32_t     Tokenizer::pad_token_id() const noexcept { return impl_->pad_id; }

std::string Tokenizer::id_to_token(int32_t id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= impl_->id_to_tok.size()) return "";
    return impl_->id_to_tok[id];
}

int32_t Tokenizer::token_to_id(const std::string& token) const {
    auto it = impl_->vocab.find(token);
    return (it != impl_->vocab.end()) ? it->second : -1;
}

} // namespace tensor::tokenizer