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

// Encode a Unicode codepoint as a UTF-8 string.
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
//  Byte ↔ Unicode mapping  (standard GPT-2 / LLaMA approach)
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
            cs.push_back(256 + n);
            ++n;
        }
    }

    std::array<std::string, 256> enc;
    for (std::size_t i = 0; i < bs.size(); ++i) {
        enc[static_cast<uint8_t>(bs[i])] = codepoint_to_utf8(cs[i]);
    }
    return enc;
}

static std::array<std::string, 256> BYTE_ENCODER = make_byte_encoder();

static std::unordered_map<std::string, uint8_t> make_byte_decoder() {
    std::unordered_map<std::string, uint8_t> dec;
    for (int b = 0; b < 256; ++b) {
        dec[BYTE_ENCODER[b]] = static_cast<uint8_t>(b);
    }
    return dec;
}

static std::unordered_map<std::string, uint8_t> BYTE_DECODER = make_byte_decoder();

// ─────────────────────────────────────────────────────────────
//  BPE encode of a single pre-tokenized word
// ─────────────────────────────────────────────────────────────

using MergeRanks = std::unordered_map<std::string, int>;

static std::vector<std::string> bpe_encode_word(
    const std::string& word_utf8,
    const MergeRanks& merge_ranks
) {
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
        int    best_rank = std::numeric_limits<int>::max();
        int    best_pos  = -1;

        for (int j = 0; j < static_cast<int>(tokens.size()) - 1; ++j) {
            std::string pair = tokens[j] + " " + tokens[j + 1];
            auto it = merge_ranks.find(pair);
            if (it != merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = j;
            }
        }

        if (best_pos == -1) break;

        std::string merged = tokens[best_pos] + tokens[best_pos + 1];
        tokens.erase(tokens.begin() + best_pos, tokens.begin() + best_pos + 2);
        tokens.insert(tokens.begin() + best_pos, merged);
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

    bool        add_bos_token   = true;
    bool        add_eos_token   = false;
    std::string chat_template;

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
    const auto& model = root["model"];
    const auto& jvocab = model["vocab"];

    std::size_t vocab_size = jvocab.size();
    id_to_tok.resize(vocab_size);

    for (const auto& [token, id_val] : jvocab.items()) {
        int32_t id = id_val.get<int32_t>();
        vocab[token] = id;
        if (static_cast<std::size_t>(id) >= id_to_tok.size()) {
            id_to_tok.resize(static_cast<std::size_t>(id) + 1);
        }
        id_to_tok[id] = token;
    }
    std::cerr << "[DEBUG tokenizer] Loaded vocab size: " << vocab.size() << "\n";

    // ── merges ───────────────────────────────────────────────
    int rank = 0;
    int array_merges = 0;
    int string_merges = 0;
    
    for (const auto& merge : model["merges"]) {
        if (merge.is_string()) {
            merge_ranks[merge.get<std::string>()] = rank++;
            string_merges++;
        } else if (merge.is_array() && merge.size() >= 2) {
            if (merge[0].is_string() && merge[1].is_string()) {
                std::string combined = merge[0].get<std::string>() + " " + merge[1].get<std::string>();
                merge_ranks[combined] = rank++;
                array_merges++;
            }
        } else {
            std::cerr << "[DEBUG tokenizer] Warning: Unrecognized BPE merge format at rank " << rank << "\n";
        }
    }
    std::cerr << "[DEBUG tokenizer] Loaded " << rank << " merges (" 
              << string_merges << " strings, " << array_merges << " arrays).\n";

    // ── added / special tokens ───────────────────────────────
    if (root.contains("added_tokens")) {
        int added_count = 0;
        for (const auto& t : root["added_tokens"]) {
            if (t.contains("content") && t["content"].is_string()) {
                const std::string content = t["content"].get<std::string>();
                const int32_t    id      = t["id"].get<int32_t>();
                special_tokens[id]      = content;
                special_vocab[content]  = id;
                vocab[content]          = id;
                if (static_cast<std::size_t>(id) >= id_to_tok.size()) {
                    id_to_tok.resize(static_cast<std::size_t>(id) + 1);
                }
                id_to_tok[id] = content;
                added_count++;
            } else {
                std::cerr << "[DEBUG tokenizer] Warning: Skipped added_token missing string 'content'\n";
            }
        }
        std::cerr << "[DEBUG tokenizer] Loaded " << added_count << " added_tokens.\n";
    }

    for (const auto& [id, str] : special_tokens) {
        if (str == "<s>" || str == "<|begin_of_text|>") {
            if (bos_id == -1) bos_id = id;
        }
        if (str == "</s>" || str == "<|end_of_text|>" || str == "<|eot_id|>") {
            if (eos_id == -1) eos_id = id;
        }
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

    auto get_token_id = [&](const std::string& key) -> int32_t {
        if (!root.contains(key)) return -1;
        const auto& v = root[key];
        
        if (v.is_number()) {
            return v.get<int32_t>();
        }
        if (v.is_array() && !v.empty() && v[0].is_number()) {
            std::cerr << "[DEBUG tokenizer] Note: '" << key << "' is an array, using first element.\n";
            return v[0].get<int32_t>();
        }
        if (v.is_object() && v.contains("content")) {
            if (v["content"].is_string()) {
                const std::string content = v["content"].get<std::string>();
                auto it = special_vocab.find(content);
                if (it != special_vocab.end()) return it->second;
            } else {
                std::cerr << "[DEBUG tokenizer] Warning: '" << key << "' object 'content' is not a string.\n";
            }
        }
        return -1;
    };

    if (int32_t id = get_token_id("bos_token_id"); id != -1) bos_id = id;
    if (int32_t id = get_token_id("eos_token_id"); id != -1) eos_id = id;
    if (int32_t id = get_token_id("pad_token_id"); id != -1) pad_id = id;

    std::cerr << "[DEBUG tokenizer] Resolved BOS: " << bos_id << ", EOS: " << eos_id << ", PAD: " << pad_id << "\n";

    if (root.contains("add_bos_token") && root["add_bos_token"].is_boolean())
        add_bos_token = root["add_bos_token"].get<bool>();
    if (root.contains("add_eos_token") && root["add_eos_token"].is_boolean())
        add_eos_token = root["add_eos_token"].get<bool>();
    
    if (root.contains("chat_template")) {
        const auto& ct = root["chat_template"];
        if (ct.is_string()) {
            chat_template = ct.get<std::string>();
        } else if (ct.is_array()) {
             std::cerr << "[DEBUG tokenizer] Note: 'chat_template' is an array, skipping automatic extraction.\n";
        }
    }
}

void Tokenizer::Impl::encode_piece(const std::string& piece,
                                    std::vector<int32_t>& out) const {
    auto it_special = special_vocab.find(piece);
    if (it_special != special_vocab.end()) {
        out.push_back(it_special->second);
        return;
    }

    std::string byte_encoded;
    for (unsigned char c : piece) {
        byte_encoded += BYTE_ENCODER[c];
    }

    auto tokens = bpe_encode_word(byte_encoded, merge_ranks);

    for (const auto& tok : tokens) {
        auto it = vocab.find(tok);
        if (it != vocab.end()) {
            out.push_back(it->second);
        }
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

    if (add_special_tokens && impl_->add_bos_token && impl_->bos_id != -1) {
        ids.push_back(impl_->bos_id);
    }

    auto words_begin = std::sregex_iterator(text.begin(), text.end(),
                                             impl_->pre_tokenize_regex);
    auto words_end   = std::sregex_iterator();

    for (auto it = words_begin; it != words_end; ++it) {
        impl_->encode_piece(it->str(), ids);
    }

    if (add_special_tokens && impl_->add_eos_token && impl_->eos_id != -1) {
        ids.push_back(impl_->eos_id);
    }

    return ids;
}

// ─────────────────────────────────────────────────────────────
//  decode
// ─────────────────────────────────────────────────────────────

std::string Tokenizer::decode(const std::vector<int32_t>& ids,
                               bool skip_special_tokens) const {
    std::string byte_str;

    for (int32_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= impl_->id_to_tok.size()) continue;

        if (skip_special_tokens &&
            impl_->special_tokens.count(id)) continue;

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
        if (it != BYTE_DECODER.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += ch;
        }
        i += char_len;
    }

    // FIXED: Replace SentencePiece space character (U+2581) with a standard space
    const std::string sp_space = "\xe2\x96\x81";
    std::size_t pos = 0;
    while ((pos = result.find(sp_space, pos)) != std::string::npos) {
        result.replace(pos, sp_space.length(), " ");
        pos += 1; // " " is 1 character long
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

    std::ostringstream out;

    if (is_llama3) {
        out << "<|begin_of_text|>";
        for (const auto& msg : messages) {
            out << "<|start_header_id|>" << msg.role << "<|end_header_id|>\n\n"
                << msg.content << "<|eot_id|>";
        }
        if (add_generation_prompt) {
            out << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }
    } else {
        for (const auto& msg : messages) {
            out << msg.role << ": " << msg.content << "\n";
        }
        if (add_generation_prompt) {
            out << "assistant: ";
        }
    }

    return out.str();
}

// ─────────────────────────────────────────────────────────────
//  vocabulary accessors
// ─────────────────────────────────────────────────────────────

std::size_t Tokenizer::vocab_size()  const noexcept { return impl_->id_to_tok.size(); }
int32_t Tokenizer::bos_token_id()    const noexcept { return impl_->bos_id; }
int32_t Tokenizer::eos_token_id()    const noexcept { return impl_->eos_id; }
int32_t Tokenizer::pad_token_id()    const noexcept { return impl_->pad_id; }

std::string Tokenizer::id_to_token(int32_t id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= impl_->id_to_tok.size()) return "";
    return impl_->id_to_tok[id];
}

int32_t Tokenizer::token_to_id(const std::string& token) const {
    auto it = impl_->vocab.find(token);
    return (it != impl_->vocab.end()) ? it->second : -1;
}

} // namespace tensor::tokenizer