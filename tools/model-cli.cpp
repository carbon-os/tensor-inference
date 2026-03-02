#include <transformers/resolver/resolver.hpp>
#include <iostream>
#include <string>
#include <cstdlib>
#include <optional>

using namespace transformers;

static void usage(const char* name) {
    std::cerr << "usage: " << name << " fetch <uri> [--no-progress]\n\n"
              << "commands:\n"
              << "  fetch <uri>    Resolve and download a model.\n"
              << "                 Example: hf://meta-llama/Llama-3.2-1B\n\n"
              << "options:\n"
              << "  --no-progress  Disable the download progress bar\n\n"
              << "note:\n"
              << "  Set the HF_TOKEN environment variable to download gated models.\n"
              << "  Models are saved to ~/.cache/models/ by default.\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    std::string command      = argv[1];
    std::string uri          = argv[2];
    bool        show_progress = true;

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-progress")
            show_progress = false;
    }

    if (command != "fetch") {
        std::cerr << "Unknown command: " << command << "\n";
        usage(argv[0]);
        return 1;
    }

    try {
        std::optional<std::string> token;
        if (const char* env_tok = std::getenv("HF_TOKEN"))
            token = env_tok;

        auto path = resolver::Resolver::fetch(uri, token, show_progress);

        std::cout << "\n[Success] Model is ready at:\n" << path.string() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "\nfatal: " << e.what() << "\n";
        return 1;
    }

    return 0;
}