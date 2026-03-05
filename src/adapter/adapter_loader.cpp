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
    m.domain               = j.value("domain",       "");
    m.architecture         = j.value("architecture", "");
    m.base_model           = j.value("base_model",   "");
    m.base_sha             = j.value("base_sha",     "");
    m.rank                 = j.value("rank",          0);
    m.alpha                = j.value("alpha",         0.0f);
    m.target_begin         = j.value("target_begin",  std::size_t{0});
    m.target_end           = j.value("target_end",    std::size_t{0});
    m.inject_q             = j.value("inject_q",     false);
    m.inject_k             = j.value("inject_k",     false);
    m.inject_v             = j.value("inject_v",     false);
    m.inject_o             = j.value("inject_o",     false);
    m.inject_up            = j.value("inject_up",    false);
    m.inject_down          = j.value("inject_down",  false);
    m.centroid_count       = j.value("centroid_count", std::size_t{0});
    m.tensor_adapt_version = j.value("tensor_adapt_version", "");

    return m;
}

// ─────────────────────────────────────────────────────────────
//  ensure_2d
//
//  Adapters serialized by tensor_adapt store LoRA weights as
//  flat 1D arrays — the 2D shape is dropped at export time.
//  We reconstruct it from the known rank:
//
//    lora_A : [rank, in_features]   → total = rank * in_features
//    lora_B : [out_features, rank]  → total = out_features * rank
//
//  If the view is already 2D we return it unchanged.
// ─────────────────────────────────────────────────────────────

static core::TensorView ensure_2d_A(core::TensorView v, int rank) {
    if (v.shape.rank() != 1) return v;   // already 2D — nothing to do

    const std::size_t total       = v.shape[0];
    const std::size_t r           = static_cast<std::size_t>(rank);

    if (total % r != 0)
        throw std::runtime_error(
            "AdapterLoader: lora_A flat size " + std::to_string(total) +
            " is not divisible by rank " + std::to_string(rank));

    v.shape = core::Shape({r, total / r});   // [rank, in_features]
    return v;
}

static core::TensorView ensure_2d_B(core::TensorView v, int rank) {
    if (v.shape.rank() != 1) return v;

    const std::size_t total       = v.shape[0];
    const std::size_t r           = static_cast<std::size_t>(rank);

    if (total % r != 0)
        throw std::runtime_error(
            "AdapterLoader: lora_B flat size " + std::to_string(total) +
            " is not divisible by rank " + std::to_string(rank));

    v.shape = core::Shape({total / r, r});   // [out_features, rank]
    return v;
}

// Try to load a projection pair; returns nullopt when either key is absent.
static std::optional<LoraProjection> try_load_proj(
    const parser::WeightMap& wm,
    const std::string&       key_A,
    const std::string&       key_B,
    const backend::Device&   device,
    int                      rank)
{
    if (!wm.contains(key_A) || !wm.contains(key_B))
        return std::nullopt;

    LoraProjection proj;
    proj.lora_A = backend::Tensor::from_view(
        ensure_2d_A(wm.tensor(key_A), rank), device);
    proj.lora_B = backend::Tensor::from_view(
        ensure_2d_B(wm.tensor(key_B), rank), device);

    // Sanity check: inner dimensions must agree.
    // lora_A: [rank, K],  lora_B: [out, rank]
    // matmul_t(x, A)=[seq,rank], matmul_t(·, B)=[seq,out] — no shared dim to
    // validate here beyond rank equality, which the reshape already guarantees.
    if (proj.lora_A.shape()[0] != static_cast<std::size_t>(rank) ||
        proj.lora_B.shape()[1] != static_cast<std::size_t>(rank))
    {
        throw std::runtime_error(
            "AdapterLoader: rank mismatch after reshape for " + key_A);
    }

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

    if (meta.architecture != expected_arch)
        throw std::runtime_error(
            "AdapterLoader: architecture mismatch — adapter is '"
            + meta.architecture + "', expected '" + expected_arch + "'");

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
    aw.layers.resize(num_layers);   // layers outside target range stay empty

    const int rank = meta.rank;

    for (std::size_t i = meta.target_begin; i <= meta.target_end; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        AdapterLayer& al    = aw.layers[i];

        if (meta.inject_q)
            al.q_proj = try_load_proj(wm,
                p + "self_attn.q_proj.lora_A",
                p + "self_attn.q_proj.lora_B", device, rank);

        if (meta.inject_k)
            al.k_proj = try_load_proj(wm,
                p + "self_attn.k_proj.lora_A",
                p + "self_attn.k_proj.lora_B", device, rank);

        if (meta.inject_v)
            al.v_proj = try_load_proj(wm,
                p + "self_attn.v_proj.lora_A",
                p + "self_attn.v_proj.lora_B", device, rank);

        if (meta.inject_o)
            al.o_proj = try_load_proj(wm,
                p + "self_attn.o_proj.lora_A",
                p + "self_attn.o_proj.lora_B", device, rank);

        if (meta.inject_up) {
            al.gate_proj = try_load_proj(wm,
                p + "mlp.gate_proj.lora_A",
                p + "mlp.gate_proj.lora_B", device, rank);
            al.up_proj = try_load_proj(wm,
                p + "mlp.up_proj.lora_A",
                p + "mlp.up_proj.lora_B", device, rank);
        }

        if (meta.inject_down)
            al.down_proj = try_load_proj(wm,
                p + "mlp.down_proj.lora_A",
                p + "mlp.down_proj.lora_B", device, rank);

        // Log first layer shapes so you can verify in the output.
        if (i == meta.target_begin && al.q_proj.has_value()) {
            std::cout << "[adapter] layer " << i << " q_proj"
                      << "  A=" << al.q_proj->lora_A.shape()[0]
                      << "×"   << al.q_proj->lora_A.shape()[1]
                      << "  B=" << al.q_proj->lora_B.shape()[0]
                      << "×"   << al.q_proj->lora_B.shape()[1] << "\n";
        }
    }

    std::cout << "[adapter] loaded — scale = "
              << aw.scale() << "  (" << meta.alpha
              << " / " << meta.rank << ")\n";

    return aw;
}

} // namespace tensor::adapter