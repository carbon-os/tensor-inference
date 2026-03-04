/*
 * resolver.cpp — HuggingFace model resolver & downloader
 *
 * HTTP/SSL NOTES:
 * ---------------
 * We use libcurl instead of cpp-httplib. cpp-httplib requires manual SSL
 * certificate configuration and has fragile redirect/SSL behaviour on Linux.
 * libcurl handles SSL via the system OpenSSL + CA bundle automatically,
 * correctly strips Authorization on cross-host redirects, and is the most
 * battle-tested HTTP library in existence.
 *
 * HF REDIRECT NOTES:
 * ------------------
 * HuggingFace serves large files (safetensors, etc.) via XetHub CAS
 * (cas-bridge.xethub.hf.co). The resolve/main/{file} endpoint returns a
 * 302 to a pre-signed URL on that host. Pre-signed URLs must NOT carry an
 * Authorization header.
 *
 * We use a two-step approach:
 *   Step 1 — fetch_metadata(): HEAD-like GET Range:bytes=0-0 to HF only,
 *             redirect following disabled. Reads x-repo-commit, etag, size,
 *             and the Location header (the real CDN download URL).
 *   Step 2 — download_file(): GET directly to the resolved CDN URL with no
 *             Authorization header.
 */

#include <tensor/resolver/resolver.hpp>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor::resolver {

std::optional<std::filesystem::path> Resolver::custom_cache_dir_ = std::nullopt;

// ---------------------------------------------------------------------------
// CURL RAII wrapper
// ---------------------------------------------------------------------------

struct CurlHandle {
    CURL* h;
    explicit CurlHandle() : h(curl_easy_init()) {
        if (!h) throw std::runtime_error("curl_easy_init() failed");
    }
    ~CurlHandle() { curl_easy_cleanup(h); }
    CurlHandle(const CurlHandle&)            = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;
    operator CURL*() const { return h; }
};

struct CurlHeaders {
    curl_slist* list = nullptr;
    void append(const std::string& header) {
        list = curl_slist_append(list, header.c_str());
    }
    ~CurlHeaders() { if (list) curl_slist_free_all(list); }
};

// Global init — called once.
struct CurlGlobal {
    CurlGlobal()  { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobal() { curl_global_cleanup(); }
} g_curl_global;

// ---------------------------------------------------------------------------
// curl write callbacks
// ---------------------------------------------------------------------------

static size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* s = static_cast<std::string*>(userdata);
    s->append(ptr, size * nmemb);
    return size * nmemb;
}

static size_t write_to_file(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* f = static_cast<std::ofstream*>(userdata);
    f->write(ptr, static_cast<std::streamsize>(size * nmemb));
    return size * nmemb;
}

// ---------------------------------------------------------------------------
// Size formatting
// ---------------------------------------------------------------------------

static std::string fmt_size(double bytes) {
    char buf[64];
    if (bytes < 1024.0 * 1024.0)
        std::snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    else
        std::snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024.0));
    return buf;
}

// ---------------------------------------------------------------------------
// FileMetadata
// ---------------------------------------------------------------------------

struct FileMetadata {
    std::string commit_hash;
    std::string etag;
    uint64_t    size         = 0;
    std::string download_url;
};

// ---------------------------------------------------------------------------
// Cache dir
// ---------------------------------------------------------------------------

std::filesystem::path Resolver::default_cache_dir() {
    const char* home = std::getenv("HOME");
    if (!home)
        throw std::runtime_error("Resolver: HOME environment variable not set.");
    return std::filesystem::path(home) / ".cache" / "models";
}

void Resolver::set_cache_dir(std::filesystem::path path) {
    custom_cache_dir_ = std::move(path);
}

std::filesystem::path Resolver::get_cache_dir() {
    return custom_cache_dir_.value_or(default_cache_dir());
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

std::filesystem::path Resolver::fetch(const std::string& uri,
                                      const std::optional<std::string>& token,
                                      bool show_progress) {
    const std::string hf_prefix = "hf://";
    if (uri.starts_with(hf_prefix))
        return resolve_hf(uri.substr(hf_prefix.length()), token, show_progress);

    throw std::invalid_argument("Resolver: Unsupported URI schema: " + uri);
}

// ---------------------------------------------------------------------------
// HuggingFace Hub API — file list
// Works on public repos without a token.
// ---------------------------------------------------------------------------

std::vector<std::string> Resolver::fetch_file_list(const std::string& repo_id,
                                                    const std::optional<std::string>& token) {
    CurlHandle curl;
    CurlHeaders headers;
    headers.append("User-Agent: libtransformers/0.1.0");
    if (token && !token->empty())
        headers.append("Authorization: Bearer " + *token);

    std::string url  = "https://huggingface.co/api/models/" + repo_id;
    std::string body;

    curl_easy_setopt(curl, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     headers.list);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  write_to_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,      &body);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK)
        throw std::runtime_error(
            "Could not reach HuggingFace Hub: " +
            std::string(curl_easy_strerror(rc)) +
            "\nCheck your internet connection.");

    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

    if (status == 401)
        throw std::runtime_error(
            "Repository '" + repo_id + "' requires authentication.\n"
            "Set the HF_TOKEN environment variable with a valid token.");

    if (status == 404)
        throw std::runtime_error(
            "Repository '" + repo_id + "' not found on HuggingFace Hub.\n"
            "Double-check the repo name at https://huggingface.co/" + repo_id);

    if (status != 200)
        throw std::runtime_error(
            "HuggingFace Hub API returned HTTP " + std::to_string(status) +
            " for repository '" + repo_id + "'.");

    try {
        auto json = nlohmann::json::parse(body);
        std::vector<std::string> files;
        for (const auto& s : json.at("siblings"))
            files.push_back(s.at("rfilename").get<std::string>());

        if (files.empty())
            throw std::runtime_error(
                "Repository '" + repo_id + "' exists but contains no files.");

        return files;

    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(
            "Failed to parse Hub API response for '" + repo_id + "': " +
            std::string(e.what()));
    }
}

// ---------------------------------------------------------------------------
// Metadata pre-fetch
// GET Range:bytes=0-0 to HF, stop before following the CDN redirect.
// Extracts commit hash, etag, file size, and the resolved CDN download URL.
// ---------------------------------------------------------------------------

static FileMetadata fetch_metadata(const std::string& hf_url,
                                   const std::string& auth_header) {
    CurlHandle  curl;
    CurlHeaders headers;
    headers.append("User-Agent: libtransformers/0.1.0");
    headers.append("Range: bytes=0-0");
    if (!auth_header.empty())
        headers.append(auth_header);

    std::string body;
    std::string raw_headers;

    curl_easy_setopt(curl, CURLOPT_URL,             hf_url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,      headers.list);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,   write_to_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,       &body);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION,  write_to_string);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA,      &raw_headers);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,  15L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,         30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION,  0L);

    CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK)
        throw std::runtime_error("Metadata request failed: " +
                                 std::string(curl_easy_strerror(rc)));

    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

    // Surface auth errors clearly so resolve_hf can catch them by type.
    if (status == 401 || status == 403)
        throw std::runtime_error("HTTP_AUTH");

    if (status != 200 && status != 206 &&
        status != 301 && status != 302 &&
        status != 307 && status != 308)
        throw std::runtime_error("Metadata request failed: HTTP " +
                                 std::to_string(status));

    auto get_header = [&](const std::string& name) -> std::string {
        std::string lower_name = name;
        for (auto& c : lower_name) c = static_cast<char>(std::tolower(c));

        std::istringstream ss(raw_headers);
        std::string line;
        while (std::getline(ss, line)) {
            if (line.size() > name.size() + 1) {
                std::string lower_line = line.substr(0, name.size());
                for (auto& c : lower_line) c = static_cast<char>(std::tolower(c));
                if (lower_line == lower_name && line[name.size()] == ':') {
                    std::string val = line.substr(name.size() + 1);
                    size_t start = val.find_first_not_of(" \t");
                    size_t end   = val.find_last_not_of(" \t\r\n");
                    if (start == std::string::npos) return "";
                    return val.substr(start, end - start + 1);
                }
            }
        }
        return "";
    };

    FileMetadata meta;

    meta.commit_hash = get_header("x-repo-commit");
    if (meta.commit_hash.empty())
        throw std::runtime_error("Metadata missing x-repo-commit header");

    meta.etag = get_header("x-linked-etag");
    if (meta.etag.empty())
        meta.etag = get_header("etag");
    if (meta.etag.size() >= 2 && meta.etag.front() == '"')
        meta.etag = meta.etag.substr(1, meta.etag.size() - 2);

    std::string cr = get_header("content-range");
    if (cr.empty()) {
        std::string cl = get_header("content-length");
        meta.size = cl.empty() ? 0 : std::stoull(cl);
    } else {
        auto slash = cr.rfind('/');
        meta.size = (slash != std::string::npos) ? std::stoull(cr.substr(slash + 1)) : 0;
    }

    meta.download_url = get_header("location");
    if (!meta.download_url.empty()) {
        if (meta.download_url[0] == '/')
            meta.download_url = "https://huggingface.co" + meta.download_url;
    } else {
        meta.download_url = hf_url;
    }

    return meta;
}

// ---------------------------------------------------------------------------
// Progress state (shared across curl progress callback)
// ---------------------------------------------------------------------------

struct ProgressState {
    uint64_t expected_size;
    bool     show;
};

static int progress_callback(void* clientp,
                              curl_off_t dltotal,
                              curl_off_t dlnow,
                              curl_off_t /*ultotal*/,
                              curl_off_t /*ulnow*/) {
    auto* state = static_cast<ProgressState*>(clientp);
    if (!state->show) return 0;

    uint64_t total = (dltotal > 0)
                         ? static_cast<uint64_t>(dltotal)
                         : state->expected_size;
    double d = static_cast<double>(dlnow);

    if (total == 0) {
        std::cout << "\r\033[K[=>] " << fmt_size(d) << " downloaded";
        std::cout.flush();
        return 0;
    }

    constexpr int W = 50;
    float ratio = static_cast<float>(dlnow) / static_cast<float>(total);
    int   pos   = static_cast<int>(W * ratio);

    std::cout << "\r\033[K[";
    for (int i = 0; i < W; ++i) {
        if      (i < pos)  std::cout << '=';
        else if (i == pos) std::cout << '>';
        else               std::cout << ' ';
    }

    char buf[128];
    std::snprintf(buf, sizeof(buf), "] %d%% (%s / %s)",
                  static_cast<int>(ratio * 100.0f),
                  fmt_size(d).c_str(),
                  fmt_size(static_cast<double>(total)).c_str());
    std::cout << buf;
    std::cout.flush();
    return 0;
}

// ---------------------------------------------------------------------------
// Download a single file straight to disk from the resolved CDN URL.
// No Authorization header — CDN URLs are pre-signed.
// ---------------------------------------------------------------------------

static void download_file(const std::string& url,
                          const std::filesystem::path& out_path,
                          uint64_t expected_size,
                          bool show_progress) {
    CurlHandle curl;

    // No auth header — CDN pre-signed URLs reject Authorization.
    CurlHeaders headers;
    headers.append("User-Agent: libtransformers/0.1.0");

    std::ofstream out(out_path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Could not open for writing: " + out_path.string());

    ProgressState progress_state{ expected_size, show_progress };

    curl_easy_setopt(curl, CURLOPT_URL,             url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,      headers.list);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,   write_to_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,       &out);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION,  1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,  15L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,         0L);   // no timeout for large files
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS,      0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA,    &progress_state);

    CURLcode rc = curl_easy_perform(curl);
    out.close();

    if (rc != CURLE_OK)
        throw std::runtime_error("Download failed: " +
                                 std::string(curl_easy_strerror(rc)));

    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    if (status != 200 && status != 206)
        throw std::runtime_error("Download failed: HTTP " + std::to_string(status));
}

// ---------------------------------------------------------------------------
// resolve_hf
// ---------------------------------------------------------------------------

std::filesystem::path Resolver::resolve_hf(const std::string& repo_id,
                                            const std::optional<std::string>& token,
                                            bool show_progress) {
    std::filesystem::path model_dir = get_cache_dir() / repo_id;

    if (std::filesystem::exists(model_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
            if (entry.path().extension() == ".safetensors")
                return model_dir;
        }
    }

    std::filesystem::create_directories(model_dir);
    std::cout << "Resolver: Fetching " << repo_id << " to " << model_dir << "\n";

    std::vector<std::string> files = fetch_file_list(repo_id, token);

    std::string auth_header;
    if (token && !token->empty())
        auth_header = "Authorization: Bearer " + *token;

    for (const auto& file : files) {
        std::filesystem::path out_file = model_dir / file;
        std::filesystem::create_directories(out_file.parent_path());

        if (show_progress)
            std::cout << "Downloading " << file << "...\n";
        else
            std::cout << "Downloading " << file << "... " << std::flush;

        std::string hf_url =
            "https://huggingface.co/" + repo_id + "/resolve/main/" + file;

        try {
            FileMetadata meta = fetch_metadata(hf_url, auth_header);
            download_file(meta.download_url, out_file, meta.size, show_progress);

        } catch (const std::exception& e) {
            std::filesystem::remove(out_file);

            std::string err = e.what();

            if (err == "HTTP_AUTH") {
                // Clean up the empty model dir before bailing
                std::filesystem::remove_all(model_dir);
                throw std::runtime_error(
                    "Access denied to '" + repo_id + "'.\n"
                    "This is a gated model that requires authentication.\n"
                    "  1. Accept the model license at https://huggingface.co/" + repo_id + "\n"
                    "  2. Generate a token at https://huggingface.co/settings/tokens\n"
                    "  3. Re-run with: HF_TOKEN=<your_token> " +
                    "model-cli fetch hf://" + repo_id);
            }

            if (err.find("HTTP 404") != std::string::npos) {
                if (show_progress)
                    std::cout << "\r\033[K  -> Skipped (404 — file is optional)\n";
                else
                    std::cout << "Skipped (404)\n";
                continue;
            }

            throw std::runtime_error(
                "Resolver: Failed to download " + file + " (" + err + ")");
        }

        if (show_progress)
            std::cout << "\n";
        else
            std::cout << "Done.\n";
    }

    return model_dir;
}

} // namespace tensor::resolver