// ─────────────────────────────────────────────────────────────
//  SafeTensors parser exercise.
//
//  Usage:
//    ./parser_safetensors <model.safetensors>
//    ./parser_safetensors <shard1.safetensors> <shard2.safetensors> ...
//    ./parser_safetensors <model_directory/>
//
//  Prints everything the parser can tell us about the file(s)
//  so every code path is visibly exercised.
// ─────────────────────────────────────────────────────────────

#include <tensor/core/dtype.hpp>
#include <tensor/core/shape.hpp>
#include <tensor/core/tensor_view.hpp>
#include <tensor/parser/errors.hpp>
#include <tensor/parser/safetensors.hpp>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using tensor::core::DType;
using tensor::core::Shape;
using tensor::core::TensorView;
using tensor::parser::FormatError;
using tensor::parser::MetaNotFound;
using tensor::parser::ParseError;
using tensor::parser::SafeTensors;
using tensor::parser::TensorInfo;
using tensor::parser::TensorNotFound;

constexpr std::size_t kMaxPrintedItems = 10;

// ─────────────────────────────────────────────────────────────
//  Formatting helpers
// ─────────────────────────────────────────────────────────────

static std::string fmt_shape(const Shape& s) {
    if (s.rank() == 0) return "[]";
    std::string out = "[";
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(s[i]);
    }
    out += "]";
    return out;
}

static std::string fmt_bytes(std::size_t n) {
    const double kb = n / 1024.0;
    const double mb = kb / 1024.0;
    const double gb = mb / 1024.0;

    std::ostringstream ss;
    ss << n << " B";
    if (gb >= 1.0)      ss << std::fixed << std::setprecision(2) << "  (" << gb << " GiB)";
    else if (mb >= 1.0) ss << std::fixed << std::setprecision(2) << "  (" << mb << " MiB)";
    else if (kb >= 1.0) ss << std::fixed << std::setprecision(2) << "  (" << kb << " KiB)";
    return ss.str();
}

static std::string fmt_offsets(std::uint64_t begin, std::uint64_t end) {
    std::ostringstream ss;
    ss << "[" << begin << ", " << end << ")";
    return ss.str();
}

static void divider(char c = '-', int width = 72) {
    std::cout << std::string(static_cast<std::size_t>(width), c) << "\n";
}

static void section(const std::string& title) {
    std::cout << "\n";
    divider();
    std::cout << "  " << title << "\n";
    divider();
}

// ─────────────────────────────────────────────────────────────
//  Open helper — picks single / multi-file / directory
// ─────────────────────────────────────────────────────────────

static SafeTensors open_from_args(int argc, char** argv) {
    if (argc == 2) {
        const std::string arg = argv[1];
        if (fs::is_directory(arg)) {
            std::cout << "[open]  directory:  " << arg << "\n";
            return SafeTensors::open_dir(arg);
        } else {
            std::cout << "[open]  single file: " << arg << "\n";
            return SafeTensors::open(arg);
        }
    }

    std::vector<std::string> paths;
    paths.reserve(static_cast<std::size_t>(argc - 1));
    std::cout << "[open]  sharded (" << (argc - 1) << " files):\n";
    for (int i = 1; i < argc; ++i) {
        std::cout << "          " << argv[i] << "\n";
        paths.push_back(argv[i]);
    }
    return SafeTensors::open(paths);
}

// ─────────────────────────────────────────────────────────────
//  Section 1 — __metadata__
// ─────────────────────────────────────────────────────────────

static void dump_metadata(const SafeTensors& st) {
    section("METADATA  (__metadata__ block)");

    const auto& meta = st.metadata();
    if (meta.empty()) {
        std::cout << "  (none)\n";
    } else {
        std::vector<std::string> keys;
        keys.reserve(meta.size());
        for (const auto& [k, _] : meta) keys.push_back(k);
        std::sort(keys.begin(), keys.end());

        std::size_t max_key = 0;
        for (const auto& k : keys) max_key = std::max(max_key, k.size());

        for (const auto& k : keys) {
            std::cout << "  "
                      << std::left << std::setw(static_cast<int>(max_key + 2)) << k
                      << st.metadata(k) << "\n";
        }
    }

    // ── has_metadata ─────────────────────────────────────────
    std::cout << "\n[has_metadata]\n";
    const std::vector<std::string> probe_keys = {"format", "version", "__metadata__", ""};
    for (const auto& k : probe_keys) {
        std::cout << "  has_metadata(\"" << k << "\") = "
                  << (st.has_metadata(k) ? "true" : "false") << "\n";
    }

    // ── metadata(key, default) ────────────────────────────────
    std::cout << "\n[metadata with default]\n";
    std::cout << "  metadata(\"format\",  \"<missing>\") = "
              << st.metadata("format",  "<missing>") << "\n";
    std::cout << "  metadata(\"version\", \"<missing>\") = "
              << st.metadata("version", "<missing>") << "\n";
    std::cout << "  metadata(\"__no_such_key__\", \"fallback\") = "
              << st.metadata("__no_such_key__", "fallback") << "\n";

    // ── metadata(key) throws MetaNotFound ────────────────────
    std::cout << "\n[metadata throw on missing key]\n";
    try {
        (void)st.metadata("__no_such_key__");
        std::cout << "  ERROR: expected MetaNotFound, got no exception\n";
    } catch (const transformers::parser::MetaNotFound& e) {
        std::cout << "  caught MetaNotFound (expected): " << e.what() << "\n";
    }
}

// ─────────────────────────────────────────────────────────────
//  Section 2 — inventory overview
// ─────────────────────────────────────────────────────────────

static void dump_inventory(const SafeTensors& st) {
    section("INVENTORY");

    std::cout << "  size()        = " << st.size()        << " tensors\n";
    std::cout << "  total_bytes() = " << fmt_bytes(st.total_bytes()) << "\n";

    // ── contains ─────────────────────────────────────────────
    std::cout << "\n[contains]\n";

    const auto names = st.names();
    if (!names.empty()) {
        const std::string& real = names.front();
        std::cout << "  contains(\"" << real << "\") = "
                  << (st.contains(real) ? "true" : "false") << "\n";
    }
    std::cout << "  contains(\"__no_such_tensor__\") = "
              << (st.contains("__no_such_tensor__") ? "true" : "false") << "\n";
}

// ─────────────────────────────────────────────────────────────
//  Section 3 — per-tensor info (TensorInfo, no data pointer)
// ─────────────────────────────────────────────────────────────

static void dump_tensor_info(const SafeTensors& st) {
    section("TENSOR INFO  (header only, no data touched)");

    const auto names = st.names();
    if (names.empty()) {
        std::cout << "  (no tensors)\n";
        return;
    }

    auto sorted = names;
    std::sort(sorted.begin(), sorted.end());

    std::size_t max_name  = 0;
    std::size_t max_dtype = 0;
    std::size_t max_shape = 0;
    for (const auto& n : sorted) {
        const TensorInfo& info = st.info(n);
        max_name  = std::max(max_name,  n.size());
        max_dtype = std::max(max_dtype, to_string(info.dtype).size());
        max_shape = std::max(max_shape, fmt_shape(info.shape).size());
    }

    std::cout << "  "
              << std::left  << std::setw(static_cast<int>(max_name  + 2)) << "name"
              << std::left  << std::setw(static_cast<int>(max_dtype + 2)) << "dtype"
              << std::left  << std::setw(static_cast<int>(max_shape + 2)) << "shape"
              << std::right << std::setw(10) << "numel"
              << std::right << std::setw(16) << "nbytes"
              << "  offsets\n";
    divider('-');

    std::size_t printed = 0;
    for (const auto& n : sorted) {
        if (printed >= kMaxPrintedItems) {
            std::cout << "  ... (" << (sorted.size() - kMaxPrintedItems) << " more tensors omitted)\n";
            break;
        }
        const TensorInfo& info = st.info(n);
        std::cout << "  "
                  << std::left  << std::setw(static_cast<int>(max_name  + 2)) << n
                  << std::left  << std::setw(static_cast<int>(max_dtype + 2)) << to_string(info.dtype)
                  << std::left  << std::setw(static_cast<int>(max_shape + 2)) << fmt_shape(info.shape)
                  << std::right << std::setw(10) << info.numel()
                  << std::right << std::setw(16) << info.nbytes()
                  << "  " << fmt_offsets(info.offset_begin, info.offset_end)
                  << "\n";
        ++printed;
    }

    // ── info() rank/numel/nbytes helpers on first tensor ─────
    const TensorInfo& first = st.info(sorted.front());
    std::cout << "\n[TensorInfo helpers on \"" << sorted.front() << "\"]\n";
    std::cout << "  rank()   = " << first.rank()   << "\n";
    std::cout << "  numel()  = " << first.numel()  << "\n";
    std::cout << "  nbytes() = " << first.nbytes() << "\n";

    // ── info() throws TensorNotFound ─────────────────────────
    std::cout << "\n[info() throw on missing tensor]\n";
    try {
        (void)st.info("__no_such_tensor__");
        std::cout << "  ERROR: expected TensorNotFound, got no exception\n";
    } catch (const transformers::parser::TensorNotFound& e) {
        std::cout << "  caught TensorNotFound (expected): " << e.what() << "\n";
    }
}

// ─────────────────────────────────────────────────────────────
//  Section 4 — TensorView (data pointer + typed span)
// ─────────────────────────────────────────────────────────────

static void dump_tensor_views(const SafeTensors& st) {
    section("TENSOR VIEWS  (zero-copy data pointers)");

    const auto names = st.names();
    if (names.empty()) {
        std::cout << "  (no tensors)\n";
        return;
    }

    auto sorted = names;
    std::sort(sorted.begin(), sorted.end());

    std::size_t printed = 0;
    for (const auto& n : sorted) {
        if (printed >= kMaxPrintedItems) {
            std::cout << "  ... (" << (sorted.size() - kMaxPrintedItems) << " more tensors omitted)\n";
            break;
        }

        TensorView view = st.tensor(n);

        std::cout << "  [" << n << "]\n";
        std::cout << "    data    = " << view.data   << "  (mmap ptr)\n";
        std::cout << "    dtype   = " << to_string(view.dtype) << "\n";
        std::cout << "    shape   = " << fmt_shape(view.shape) << "\n";
        std::cout << "    rank()  = " << view.rank()  << "\n";
        std::cout << "    numel() = " << view.numel() << "\n";
        std::cout << "    nbytes()= " << view.nbytes() << "\n";

        {
            const auto* bytes = static_cast<const std::uint8_t*>(view.data);
            const std::size_t total = view.nbytes();
            const std::size_t peek  = std::min(total, std::size_t{8});

            std::cout << "    first " << peek << " raw bytes = [ ";
            for (std::size_t i = 0; i < peek; ++i) {
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<unsigned>(bytes[i]) << " ";
            }
            std::cout << std::dec << std::setfill(' ') << "]\n";
        }

        switch (view.dtype) {
            case DType::F64: {
                auto sp = view.as<double>();
                std::cout << "    as<double>() span size = " << sp.size() << "\n";
                break;
            }
            case DType::F32: {
                auto sp = view.as<float>();
                std::cout << "    as<float>()  span size = " << sp.size() << "\n";
                break;
            }
            case DType::BF16:
            case DType::F16: {
                auto sp = view.as<std::uint16_t>();
                std::cout << "    as<uint16>() span size = " << sp.size() << "\n";
                break;
            }
            case DType::F8_E5M2:
            case DType::F8_E4M3:
            case DType::I8:
            case DType::U8:
            case DType::BOOL: {
                auto sp = view.as<std::uint8_t>();
                std::cout << "    as<uint8>()  span size = " << sp.size() << "\n";
                break;
            }
            case DType::I64: {
                auto sp = view.as<std::int64_t>();
                std::cout << "    as<int64>()  span size = " << sp.size() << "\n";
                break;
            }
            case DType::I32: {
                auto sp = view.as<std::int32_t>();
                std::cout << "    as<int32>()  span size = " << sp.size() << "\n";
                break;
            }
            case DType::I16: {
                auto sp = view.as<std::int16_t>();
                std::cout << "    as<int16>()  span size = " << sp.size() << "\n";
                break;
            }
        }

        std::cout << "\n";
        ++printed;
    }

    // ── as<T>() throws on mismatched size ────────────────────
    {
        std::cout << "[as<T>() throw on size mismatch]\n";
        bool tested = false;
        for (const auto& n : sorted) {
            TensorView view = st.tensor(n);
            if (transformers::core::dtype_size(view.dtype) == 2) {
                try {
                    (void)view.as<float>();
                    std::cout << "  ERROR: expected invalid_argument, got no exception\n";
                } catch (const std::invalid_argument& e) {
                    std::cout << "  caught invalid_argument on \"" << n
                              << "\" (expected): " << e.what() << "\n";
                }
                tested = true;
                break;
            }
        }
        if (!tested) {
            std::cout << "  (skipped — no 2-byte dtype tensor found to test mismatch)\n";
        }
    }

    // ── tensor() throws TensorNotFound ───────────────────────
    std::cout << "\n[tensor() throw on missing name]\n";
    try {
        (void)st.tensor("__no_such_tensor__");
        std::cout << "  ERROR: expected TensorNotFound, got no exception\n";
    } catch (const transformers::parser::TensorNotFound& e) {
        std::cout << "  caught TensorNotFound (expected): " << e.what() << "\n";
    }
}

// ─────────────────────────────────────────────────────────────
//  Section 5 — tensors() iteration
// ─────────────────────────────────────────────────────────────

static void dump_tensors_iteration(const SafeTensors& st) {
    section("TENSORS()  (iteration over all (name, TensorView) pairs)");

    const auto tmap = st.tensors();

    std::cout << "  tensors() returned " << tmap.size() << " entries"
              << "  (matches size()=" << st.size() << ": "
              << (tmap.size() == st.size() ? "OK" : "MISMATCH") << ")\n\n";

    std::vector<std::string> keys;
    keys.reserve(tmap.size());
    for (const auto& [k, _] : tmap) keys.push_back(k);
    std::sort(keys.begin(), keys.end());

    std::size_t printed = 0;
    for (const auto& name : keys) {
        if (printed >= kMaxPrintedItems) {
            std::cout << "  ... (" << (keys.size() - kMaxPrintedItems) << " more tensors omitted)\n";
            break;
        }
        const TensorView& view = tmap.at(name);
        std::cout << "  " << name
                  << "  dtype=" << to_string(view.dtype)
                  << "  shape=" << fmt_shape(view.shape)
                  << "  nbytes=" << view.nbytes()
                  << "  ptr=" << view.data
                  << "\n";
        ++printed;
    }

    std::cout << "\n[cross-check: names() ⊆ tensors()]\n";
    bool all_found = true;
    for (const auto& n : st.names()) {
        if (!tmap.count(n)) {
            std::cout << "  MISSING: \"" << n << "\"\n";
            all_found = false;
        }
    }
    if (all_found) {
        std::cout << "  all " << st.names().size() << " names present in tensors() map: OK\n";
    }
}

// ─────────────────────────────────────────────────────────────
//  Section 6 — summary statistics
// ─────────────────────────────────────────────────────────────

static void dump_summary(const SafeTensors& st) {
    section("SUMMARY");

    std::unordered_map<std::string, std::size_t> dtype_count;
    std::unordered_map<std::string, std::size_t> dtype_bytes;
    std::size_t max_tensor_bytes = 0;
    std::string max_tensor_name;
    std::size_t scalar_count  = 0;
    std::size_t vector_count  = 0;
    std::size_t matrix_count  = 0;
    std::size_t higher_count  = 0;

    for (const auto& name : st.names()) {
        const TensorInfo& info = st.info(name);
        const std::string dstr = to_string(info.dtype);

        dtype_count[dstr]++;
        dtype_bytes[dstr] += info.nbytes();

        if (info.nbytes() > max_tensor_bytes) {
            max_tensor_bytes = info.nbytes();
            max_tensor_name  = name;
        }

        switch (info.rank()) {
            case 0:  scalar_count++;  break;
            case 1:  vector_count++;  break;
            case 2:  matrix_count++;  break;
            default: higher_count++;  break;
        }
    }

    std::cout << "  Total tensors : " << st.size() << "\n";
    std::cout << "  Total bytes   : " << fmt_bytes(st.total_bytes()) << "\n";

    std::cout << "\n  Rank distribution:\n";
    std::cout << "    rank-0 (scalar) : " << scalar_count << "\n";
    std::cout << "    rank-1 (vector) : " << vector_count << "\n";
    std::cout << "    rank-2 (matrix) : " << matrix_count << "\n";
    std::cout << "    rank-3+         : " << higher_count << "\n";

    std::cout << "\n  DType breakdown:\n";
    std::vector<std::string> dtypes;
    for (const auto& [k, _] : dtype_count) dtypes.push_back(k);
    std::sort(dtypes.begin(), dtypes.end());
    for (const auto& d : dtypes) {
        std::cout << "    " << std::left << std::setw(10) << d
                  << " tensors=" << std::setw(6) << dtype_count[d]
                  << "  bytes=" << fmt_bytes(dtype_bytes[d]) << "\n";
    }

    if (!max_tensor_name.empty()) {
        std::cout << "\n  Largest tensor  : \"" << max_tensor_name
                  << "\"  " << fmt_bytes(max_tensor_bytes) << "\n";
    }
}

// ─────────────────────────────────────────────────────────────
//  Section 7 — error paths
// ─────────────────────────────────────────────────────────────

static void test_error_paths() {
    section("ERROR PATHS");

    std::cout << "[open non-existent file]\n";
    try {
        auto bad = SafeTensors::open("/tmp/__no_such_file_at_all__.safetensors");
        std::cout << "  ERROR: expected ParseError, got no exception\n";
    } catch (const ParseError& e) {
        std::cout << "  caught ParseError (expected): " << e.what() << "\n";
    }

    std::cout << "\n[open_dir on non-existent directory]\n";
    try {
        auto bad = SafeTensors::open_dir("/tmp/__no_such_dir_at_all__/");
        std::cout << "  ERROR: expected ParseError, got no exception\n";
    } catch (const ParseError& e) {
        std::cout << "  caught ParseError (expected): " << e.what() << "\n";
    }

    std::cout << "\n[open empty path list]\n";
    try {
        auto bad = SafeTensors::open(std::vector<std::string>{});
        std::cout << "  ERROR: expected ParseError, got no exception\n";
    } catch (const ParseError& e) {
        std::cout << "  caught ParseError (expected): " << e.what() << "\n";
    }
}

// ─────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "usage:\n"
            << "  " << argv[0] << " <model.safetensors>\n"
            << "  " << argv[0] << " <shard1.safetensors> <shard2.safetensors> ...\n"
            << "  " << argv[0] << " <model_directory/>\n";
        return 1;
    }

    divider('=');
    std::cout << "  libtransformers -- SafeTensors parser\n";
    divider('=');

    SafeTensors st = [&]() -> SafeTensors {
        try {
            return open_from_args(argc, argv);
        } catch (const ParseError& e) {
            std::cerr << "\nFATAL ParseError: " << e.what() << "\n";
            std::exit(1);
        } catch (const FormatError& e) {
            std::cerr << "\nFATAL FormatError: " << e.what() << "\n";
            std::exit(1);
        }
    }();

    dump_metadata(st);
    dump_inventory(st);
    dump_tensor_info(st);
    dump_tensor_views(st);
    dump_tensors_iteration(st);
    dump_summary(st);
    test_error_paths();

    std::cout << "\n";
    divider('=');
    std::cout << "  done\n";
    divider('=');
    std::cout << "\n";

    return 0;
}