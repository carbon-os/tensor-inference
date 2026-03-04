#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace tensor::resolver {

class Resolver {
public:
    // Resolves a URI (e.g., "hf://meta-llama/Llama-3.2-1B") to a local path.
    // If the model is not in the cache, it downloads it.
    static std::filesystem::path fetch(const std::string& uri,
                                       const std::optional<std::string>& auth_token = std::nullopt,
                                       bool show_progress = true);

    // Override the default cache directory (~/.cache/models)
    static void set_cache_dir(std::filesystem::path path);

    // Get the current cache directory
    static std::filesystem::path get_cache_dir();

private:
    static std::filesystem::path default_cache_dir();

    static std::filesystem::path resolve_hf(const std::string& repo_id,
                                            const std::optional<std::string>& token,
                                            bool show_progress);

    // Fetches the list of files in a HF repo via the Hub API.
    // Falls back to a hardcoded default list if the API call fails.
    static std::vector<std::string> fetch_file_list(const std::string& repo_id,
                                                    const std::optional<std::string>& token);

    // Internal state for custom cache paths
    static std::optional<std::filesystem::path> custom_cache_dir_;
};

} // namespace tensor::resolver