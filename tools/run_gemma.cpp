// ─────────────────────────────────────────────────────────────
//  run_gemma — end-to-end Gemma inference
//
//  Usage:
//    ./run_gemma <model_dir> [options]
// ─────────────────────────────────────────────────────────────

#include <transformers/backend/device.hpp>
#include <transformers/inference/generator.hpp>
#include <transformers/inference/sampling/greedy.hpp>
#include <transformers/inference/sampling/top_p.hpp>
#include <transformers/models/gemma/gemma_model.hpp>
#include <transformers/parser/config.hpp>
#include <transformers/parser/weight_map.hpp>
#include <transformers/tokenizer/tokenizer.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

using namespace transformers;

struct Args {
    std::string model_dir;
    std::string prompt;
    bool        chat        = false;
    std::size_t max_tokens  = 256;
    float       temperature = 1.0f;
    float       top_p       = 0.95f;
    bool        greedy      = false;
    std::optional<uint64_t> seed;
    int         device_idx  = 0;
    std::size_t max_seq     = 4096;
};

static void usage(const char* name) {
    std::cerr
        << "usage: " << name << " <model_dir> [options]\n"
        << "  --prompt    TEXT    prompt (required unless --chat)\n"
        << "  --chat              interactive chat mode\n"
        << "  --max-tokens N      max new tokens (default 256)\n"
        << "  --temperature F     temperature (default 1.0)\n"
        << "  --top-p F           nucleus p (default 0.95)\n"
        << "  --greedy            greedy decoding\n"
        << "  --seed N            random seed\n"
        << "  --device N          CUDA device index (default 0)\n"
        << "  --max-seq N         KV cache length (default 4096)\n";
}

static Args parse_args(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); std::exit(1); }

    Args a;
    a.model_dir = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { std::cerr << flag << " requires a value\n"; std::exit(1); }
            return argv[i];
        };
        if      (flag == "--prompt")      a.prompt      = next();
        else if (flag == "--chat")        a.chat        = true;
        else if (flag == "--max-tokens")  a.max_tokens  = std::stoull(next());
        else if (flag == "--temperature") a.temperature = std::stof(next());
        else if (flag == "--top-p")       a.top_p       = std::stof(next());
        else if (flag == "--greedy")      a.greedy      = true;
        else if (flag == "--seed")        a.seed        = std::stoull(next());
        else if (flag == "--device")      a.device_idx  = std::stoi(next());
        else if (flag == "--max-seq")     a.max_seq     = std::stoull(next());
        else { std::cerr << "unknown flag: " << flag << "\n"; std::exit(1); }
    }

    if (!a.chat && a.prompt.empty()) {
        std::cerr << "error: --prompt is required (or use --chat)\n";
        std::exit(1);
    }
    return a;
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    try {
        auto device = backend::Device::cuda(args.device_idx);
        std::cout << "[device] " << device.name()
                  << "  VRAM " << (device.memory_capacity() >> 20) << " MiB\n";

        std::cout << "[load]  model dir: " << args.model_dir << "\n";

        auto weights   = parser::WeightMap::open(args.model_dir);
        auto config    = parser::ModelConfig::from_dir(args.model_dir);
        auto tokenizer = tokenizer::Tokenizer::from_files(args.model_dir);

        std::cout << "[model] " << config.model_type()
                  << "  vocab=" << config.vocab_size()
                  << "  layers=" << config.num_hidden_layers()
                  << "  hidden=" << config.hidden_size()
                  << "  dtype=" << config.torch_dtype() << "\n";

        auto t0 = std::chrono::steady_clock::now();
        auto model = models::gemma::GemmaModel::load(weights, config, device);
        auto t1 = std::chrono::steady_clock::now();
        
        double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[load]  " << load_ms << " ms  ("
                  << (weights.total_bytes() >> 20) << " MiB)\n";

        auto gen = inference::Generator::create(model, tokenizer, args.max_seq);

        inference::Sampler sampler;
        if (args.greedy) {
            sampler = inference::sampling::Greedy{};
        } else {
            inference::sampling::TopP top_p;
            top_p.temperature = args.temperature;
            top_p.p           = args.top_p;
            top_p.seed        = args.seed;
            sampler = top_p;
        }

        const inference::GenerateOptions base_opts {
            .max_new_tokens = args.max_tokens,
            .sampler        = sampler,
            .stop_strings   = { "<eos>", "<end_of_turn>" },
            .on_token       = [&](int32_t tok) -> bool {
                std::string piece = tokenizer.decode({tok}, true);
                std::cout << piece << std::flush;
                return true;
            },
        };

        if (!args.chat) {
            std::cout << "\n--- output ---\n";

            // Manually apply the Gemma Chat Template to bypass the tokenizer bug
            std::string formatted_prompt = "<start_of_turn>user\n" + args.prompt + "<end_of_turn>\n<start_of_turn>model\n";
            
            // Encode without adding the buggy special tokens
            auto ids = tokenizer.encode(formatted_prompt, false);
            
            // Force prepend Gemma's correct BOS token (2)
            ids.insert(ids.begin(), 2);

            auto ts = std::chrono::steady_clock::now();
            auto result = gen.generate(ids, base_opts);
            auto te = std::chrono::steady_clock::now();

            double gen_ms  = std::chrono::duration<double, std::milli>(te - ts).count();
            double tok_per_s = result.tokens_generated / (gen_ms / 1000.0);

            std::cout << "\n--- stats ---\n"
                      << "  tokens:  " << result.tokens_generated << "\n"
                      << "  speed:   " << tok_per_s << " tok/s\n"
                      << "  reason:  " << result.stop_reason << "\n";
        } else {
            std::vector<tokenizer::Tokenizer::Message> history;
            std::cout << "[chat]  type 'quit' to exit\n\n";

            while (true) {
                std::cout << "user: " << std::flush;
                std::string line;
                if (!std::getline(std::cin, line) || line == "quit") break;
                if (line.empty()) continue;

                history.push_back({"user", line});

                // Manually apply the Gemma Chat Template
                std::string prompt;
                for (const auto& msg : history) {
                    prompt += "<start_of_turn>" + msg.role + "\n" + msg.content + "<end_of_turn>\n";
                }
                prompt += "<start_of_turn>model\n";

                auto ids = tokenizer.encode(prompt, false);
                ids.insert(ids.begin(), 2);

                std::cout << "assistant: " << std::flush;

                std::string assistant_reply;
                auto opts = base_opts;
                opts.on_token = [&](int32_t tok) -> bool {
                    std::string piece = tokenizer.decode({tok}, true);
                    std::cout << piece << std::flush;
                    assistant_reply += piece;
                    return true;
                };

                auto result = gen.generate(ids, opts);
                std::cout << "\n\n";

                history.push_back({"assistant", assistant_reply});
                gen.reset();
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "\nfatal: " << e.what() << "\n";
        return 1;
    }

    return 0;
}