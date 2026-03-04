#include <tensor/adapter/adapter_loader.hpp>
#include <tensor/parser/weight_map.hpp>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs   = std::filesystem;
using     json = nlohmann::json;

namespace tensor::adapter {

// ─────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────

static AdapterMeta parse_meta(const fs::path& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open())
        throw std::runtime_error("AdapterLoader: cannot open " + json_path.string());

    json j;
    try { j = json::parse(f); }
    catch (const json::exception& e) {
        throw std::runtime_error(
            std::string("AdapterLoader: adapter.json parse error: ") + e.what());
    }

    AdapterMeta m;
    m.domain                 = j.value("domain",       "");
    m.architecture           = j.value("architecture", "");
    m.base_model             = j.value("base_model",   "");
    m.base_sha               = j.value("base_sha",     "");
    m.rank                   = j.value("rank",          0);
    m.alpha                  = j.value("alpha",         0.0f);
    m.target_begin           = j.value("target_begin",  std::size_t{0});
    m.target_end             = j.value("target_end",    std::size_t{0});
    m.inject_q               = j.value("inject_q",     false);
    m.inject_k               = j.value("inject_k",     false);
    m.inject_v               = j.value("inject_v",     false);
    m.inject_o               = j.value("inject_o",     false);
    m.inject_up              = j.value("inject_up",    false);
    m.inject_down            = j.value("inject_down",  false);
    m.centroid_count         = j.value("centroid_count", std::size_t{0});
    m.tensor_adapt_version   = j.value("tensor_adapt_version", "");

    return m;
}

// Try to load a weight; returns empty optional when the key is absent.
// We use "optional" presence rather than hard-failing so the loader
// gracefully handles adapters that omit FFN projections (rank < 800M).
static std::optional<LoraProjection> try_load_proj(
    const parser::WeightMap& wm,
    const std::string&       key_A,
    const std::string&       key_B,
    const backend::Device&   device)
{
    if (!wm.contains(key_A) || !wm.contains(key_B))
        return std::nullopt;

    LoraProjection proj;
    proj.lora_A = backend::Tensor::from_view(wm.tensor(key_A), device);
    proj.lora_B = backend::Tensor::from_view(wm.tensor(key_B), device);
    return proj;
}

// ─────────────────────────────────────────────────────────────
//  AdapterLoader::load
// ─────────────────────────────────────────────────────────────

AdapterWeights AdapterLoader::load(
    const std::string&     adapter_dir,
    const std::string&     expected_arch,
    std::size_t            num_layers,
    const backend::Device& device)
{
    fs::path dir(adapter_dir);

    // ── 1. metadata ───────────────────────────────────────────

    AdapterMeta meta = parse_meta(dir / "adapter.json");

    if (meta.architecture != expected_arch) {
        throw std::runtime_error(
            "AdapterLoader: architecture mismatch — adapter is '"
            + meta.architecture + "', expected '" + expected_arch + "'");
    }

    if (meta.rank <= 0)
        throw std::runtime_error("AdapterLoader: adapter.json has invalid rank");

    if (meta.target_end >= num_layers)
        throw std::runtime_error(
            "AdapterLoader: target_end (" + std::to_string(meta.target_end)
            + ") >= base num_layers (" + std::to_string(num_layers) + ")");

    std::cout << "[adapter] domain:    " << meta.domain            << "\n"
              << "[adapter] base:      " << meta.base_model         << "\n"
              << "[adapter] arch:      " << meta.architecture        << "\n"
              << "[adapter] rank:      " << meta.rank               << "\n"
              << "[adapter] alpha:     " << meta.alpha              << "\n"
              << "[adapter] layers:    " << meta.target_begin
              << " – "                   << meta.target_end         << "\n"
              << "[adapter] inject:   "
              << (meta.inject_q    ? " Q"    : "")
              << (meta.inject_k    ? " K"    : "")
              << (meta.inject_v    ? " V"    : "")
              << (meta.inject_o    ? " O"    : "")
              << (meta.inject_up   ? " up"   : "")
              << (meta.inject_down ? " down" : "")
              << "\n";

    // ── 2. weights ────────────────────────────────────────────

    auto wm = parser::WeightMap::open((dir / "adapter.safetensors").string());

    AdapterWeights aw;
    aw.meta = meta;
    aw.layers.resize(num_layers);   // layers outside the target range stay empty

    for (std::size_t i = meta.target_begin; i <= meta.target_end; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        AdapterLayer& al    = aw.layers[i];

        // Attention projections
        if (meta.inject_q)
            al.q_proj = try_load_proj(wm,
                p + "self_attn.q_proj.lora_A",
                p + "self_attn.q_proj.lora_B", device);

        if (meta.inject_k)
            al.k_proj = try_load_proj(wm,
                p + "self_attn.k_proj.lora_A",
                p + "self_attn.k_proj.lora_B", device);

        if (meta.inject_v)
            al.v_proj = try_load_proj(wm,
                p + "self_attn.v_proj.lora_A",
                p + "self_attn.v_proj.lora_B", device);

        if (meta.inject_o)
            al.o_proj = try_load_proj(wm,
                p + "self_attn.o_proj.lora_A",
                p + "self_attn.o_proj.lora_B", device);

        // FFN projections (only present for larger models)
        if (meta.inject_up)
            al.gate_proj = try_load_proj(wm,
                p + "mlp.gate_proj.lora_A",
                p + "mlp.gate_proj.lora_B", device);

        if (meta.inject_up)
            al.up_proj = try_load_proj(wm,
                p + "mlp.up_proj.lora_A",
                p + "mlp.up_proj.lora_B", device);

        if (meta.inject_down)
            al.down_proj = try_load_proj(wm,
                p + "mlp.down_proj.lora_A",
                p + "mlp.down_proj.lora_B", device);
    }

    std::cout << "[adapter] loaded — scale = "
              << aw.scale() << "  (" << meta.alpha
              << " / " << meta.rank << ")\n";

    return aw;
}

} // namespace tensor::adapter