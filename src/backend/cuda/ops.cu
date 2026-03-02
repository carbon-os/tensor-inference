#include <transformers/backend/ops.hpp>
#include <transformers/backend/tensor.hpp>
#include <transformers/core/dtype.hpp>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace transformers::backend::ops {

// ─────────────────────────────────────────────────────────────
//  Utilities
// ─────────────────────────────────────────────────────────────

static void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
}

static void cublas_check(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(msg) + " (cuBLAS error " + std::to_string(s) + ")");
}

// Per-device cuBLAS handle (created on first use).
static cublasHandle_t get_cublas(int device_index) {
    static cublasHandle_t handles[8] = {};
    static bool           ready[8]   = {};
    if (!ready[device_index]) {
        cudaSetDevice(device_index);
        cublas_check(cublasCreate(&handles[device_index]), "cublasCreate");
        cublasSetMathMode(handles[device_index], CUBLAS_DEFAULT_MATH);
        ready[device_index] = true;
    }
    return handles[device_index];
}

// ─────────────────────────────────────────────────────────────
//  embedding — gather rows
// ─────────────────────────────────────────────────────────────

__global__ void k_embedding_bf16(
    const __nv_bfloat16* __restrict__ weight,   // [vocab, hidden]
    const int* __restrict__ indices,  // [seq_len]  (device copy)
    __nv_bfloat16* __restrict__ out,       // [seq_len, hidden]
    int hidden
) {
    int token = blockIdx.x;
    int tid   = threadIdx.x;
    int id    = indices[token];

    for (int i = tid; i < hidden; i += blockDim.x) {
        out[token * hidden + i] = weight[id * hidden + i];
    }
}

Tensor embedding(const Tensor& weight, const std::vector<int32_t>& indices) {
    const int seq_len = static_cast<int>(indices.size());
    const int hidden  = static_cast<int>(weight.shape()[1]);
    const auto& dev   = weight.device();

    // Upload index list to device.
    int* d_indices = nullptr;
    const std::size_t idx_bytes = seq_len * sizeof(int);
    dev.make_current();
    cuda_check(cudaMalloc(&d_indices, idx_bytes), "embedding: cudaMalloc indices");
    cuda_check(cudaMemcpy(d_indices, indices.data(), idx_bytes, cudaMemcpyHostToDevice),
               "embedding: H2D indices");

    Tensor out = Tensor::empty({static_cast<std::size_t>(seq_len),
                                static_cast<std::size_t>(hidden)},
                               weight.dtype(), dev);

    const int threads = std::min(hidden, 256);
    k_embedding_bf16<<<seq_len, threads>>>(
        weight.data_as<__nv_bfloat16>(),
        d_indices,
        out.data_as<__nv_bfloat16>(),
        hidden
    );

    cudaFree(d_indices);
    return out;
}

// ─────────────────────────────────────────────────────────────
//  matmul_t — A @ B^T via cuBLAS (BF16)
// ─────────────────────────────────────────────────────────────

Tensor matmul_t(const Tensor& a, const Tensor& b) {
    // A: [M, K],  B: [N, K]  → C: [M, N]
    const int M = static_cast<int>(a.shape()[0]);
    const int K = static_cast<int>(a.shape()[1]);
    const int N = static_cast<int>(b.shape()[0]);

    const auto& dev = a.device();
    dev.make_current();

    Tensor c = Tensor::empty({static_cast<std::size_t>(M),
                               static_cast<std::size_t>(N)},
                              a.dtype(), dev);

    cublasHandle_t handle = get_cublas(dev.index());

    if (a.dtype() == DType::BF16) {
        // cuBLAS column-major: C^T = B * A^T, i.e. (N×M) = (N×K)(K×M)
        // which in row-major is  C(M×N) = A(M×K) @ B^T(K×N)
        const float alpha = 1.0f, beta = 0.0f;
        cublas_check(
            cublasGemmEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,   // B^T, A
                N, M, K,
                &alpha,
                b.data_ptr(), CUDA_R_16BF, K,
                a.data_ptr(), CUDA_R_16BF, K,
                &beta,
                c.data_ptr(), CUDA_R_16BF, N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "matmul_t cublasGemmEx BF16"
        );
    } else if (a.dtype() == DType::F32) {
        const float alpha = 1.0f, beta = 0.0f;
        cublas_check(
            cublasSgemm(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                b.data_as<float>(), K,
                a.data_as<float>(), K,
                &beta,
                c.data_as<float>(), N
            ),
            "matmul_t cublasSgemm F32"
        );
    } else {
        throw std::invalid_argument("matmul_t: unsupported dtype");
    }

    return c;
}

// ─────────────────────────────────────────────────────────────
//  rms_norm
// ─────────────────────────────────────────────────────────────

__global__ void k_rms_norm_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int hidden, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Accumulate sum of squares in F32
    float ss = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = __bfloat162float(x[row * hidden + i]);
        ss += v * v;
    }

    // Block reduction in shared memory
    __shared__ float smem[256];
    smem[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = __bfloat162float(x[row * hidden + i]);
        float weight = __bfloat162float(w[i]);
        out[row * hidden + i] = __float2bfloat16(v * scale * weight);
    }
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    const int seq_len = static_cast<int>(x.shape()[0]);
    const int hidden  = static_cast<int>(x.shape()[1]);

    x.device().make_current();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());

    // Launch with a fixed power-of-two (256) for safe reduction
    const int threads = 256;
    k_rms_norm_bf16<<<seq_len, threads>>>(
        x.data_as<__nv_bfloat16>(),
        weight.data_as<__nv_bfloat16>(),
        out.data_as<__nv_bfloat16>(),
        hidden, eps
    );
    return out;
}

// ─────────────────────────────────────────────────────────────
//  rope — Rotary Position Embedding (Llama-3 scaling aware)
// ─────────────────────────────────────────────────────────────

__global__ void k_rope_bf16(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ freqs,  // [head_dim/2] scaled angular freqs
    int n_heads, int head_dim, int seq_offset
) {
    int pos  = blockIdx.x;
    int head = blockIdx.y;
    int half = head_dim / 2;
    int tid  = threadIdx.x;   // 0 .. half-1

    if (tid >= half) return;

    int   base  = (pos * n_heads + head) * head_dim;
    float angle = static_cast<float>(pos + seq_offset) * freqs[tid];
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = __bfloat162float(x[base + tid]);
    float x1 = __bfloat162float(x[base + tid + half]);

    out[base + tid]        = __float2bfloat16(x0 * cos_a - x1 * sin_a);
    out[base + tid + half] = __float2bfloat16(x0 * sin_a + x1 * cos_a);
}

static std::vector<float> compute_rope_freqs(int head_dim, float theta,
                                              const RopeScaling& sc) {
    const int half = head_dim / 2;
    std::vector<float> freqs(half);

    // Wavelength thresholds that define the three regimes.
    const float pi2 = 2.0f * static_cast<float>(M_PI);
    const float low_wavelen  = static_cast<float>(sc.original_max_pos)
                             / sc.low_freq_factor;
    const float high_wavelen = static_cast<float>(sc.original_max_pos)
                             / sc.high_freq_factor;

    for (int i = 0; i < half; ++i) {
        // Base frequency for this pair: 1 / (theta^(2i/d))
        float base_freq = 1.0f / std::pow(theta,
            (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));

        if (!sc.apply_scaling) {
            freqs[i] = base_freq;
            continue;
        }

        float wavelen = pi2 / base_freq;

        float scaled_freq;
        if (wavelen < high_wavelen) {
            // High-frequency: local, no scaling.
            scaled_freq = base_freq;
        } else if (wavelen > low_wavelen) {
            // Low-frequency: global, scale down.
            scaled_freq = base_freq / sc.factor;
        } else {
            // Transition: smooth linear blend.
            float smooth = (static_cast<float>(sc.original_max_pos) / wavelen
                           - sc.low_freq_factor)
                         / (sc.high_freq_factor - sc.low_freq_factor);
            scaled_freq = (1.0f - smooth) * (base_freq / sc.factor)
                        + smooth * base_freq;
        }

        freqs[i] = scaled_freq;
    }
    return freqs;
}

Tensor rope(const Tensor& x, int seq_offset, float theta,
            const RopeScaling& scaling) {
    const int seq_len  = static_cast<int>(x.shape()[0]);
    const int n_heads  = static_cast<int>(x.shape()[1]);
    const int head_dim = static_cast<int>(x.shape()[2]);
    const int half     = head_dim / 2;

    std::vector<float> h_freqs = compute_rope_freqs(head_dim, theta, scaling);

    float* d_freqs = nullptr;
    x.device().make_current();
    cuda_check(cudaMalloc(&d_freqs, half * sizeof(float)),
               "rope: cudaMalloc freqs");
    cuda_check(cudaMemcpy(d_freqs, h_freqs.data(), half * sizeof(float),
                          cudaMemcpyHostToDevice), "rope: H2D freqs");

    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());

    dim3 grid(seq_len, n_heads);
    k_rope_bf16<<<grid, half>>>(
        x.data_as<__nv_bfloat16>(),
        out.data_as<__nv_bfloat16>(),
        d_freqs,
        n_heads, head_dim, seq_offset
    );

    cudaFree(d_freqs);
    return out;
}

Tensor rope(const Tensor& x, int seq_offset, float theta) {
    return rope(x, seq_offset, theta, RopeScaling{});
}

// ─────────────────────────────────────────────────────────────
//  silu, mul, add
// ─────────────────────────────────────────────────────────────

__global__ void k_silu_bf16(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    out[i]  = __float2bfloat16(v / (1.0f + expf(-v)));
}

Tensor silu(const Tensor& x) {
    x.device().make_current();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    const int n = static_cast<int>(x.numel());
    k_silu_bf16<<<(n + 255) / 256, 256>>>(
        x.data_as<__nv_bfloat16>(),
        out.data_as<__nv_bfloat16>(), n);
    return out;
}

__global__ void k_mul_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(__bfloat162float(a[i]) * __bfloat162float(b[i]));
}

Tensor mul(const Tensor& a, const Tensor& b) {
    a.device().make_current();
    Tensor out = Tensor::empty(a.shape(), a.dtype(), a.device());
    const int n = static_cast<int>(a.numel());
    k_mul_bf16<<<(n + 255) / 256, 256>>>(
        a.data_as<__nv_bfloat16>(),
        b.data_as<__nv_bfloat16>(),
        out.data_as<__nv_bfloat16>(), n);
    return out;
}

__global__ void k_add_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

Tensor add(const Tensor& a, const Tensor& b) {
    a.device().make_current();
    Tensor out = Tensor::empty(a.shape(), a.dtype(), a.device());
    const int n = static_cast<int>(a.numel());
    k_add_bf16<<<(n + 255) / 256, 256>>>(
        a.data_as<__nv_bfloat16>(),
        b.data_as<__nv_bfloat16>(),
        out.data_as<__nv_bfloat16>(), n);
    return out;
}

// ─────────────────────────────────────────────────────────────
//  softmax (along last dim)
// ─────────────────────────────────────────────────────────────

__global__ void k_softmax_f32(float* x, int cols) {
    // One block per row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float* row_ptr = x + row * cols;

    // Max for numerical stability
    __shared__ float smem[256];
    float local_max = -1e38f;
    for (int i = tid; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, row_ptr[i]);
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float gmax = smem[0];

    // Sum of exp
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        local_sum += expf(row_ptr[i] - gmax);
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];

    for (int i = tid; i < cols; i += blockDim.x)
        row_ptr[i] = expf(row_ptr[i] - gmax) * inv_sum;
}

Tensor softmax(const Tensor& x) {
    // Promote to F32 for softmax, return F32
    x.device().make_current();
    // Cast to F32 first if needed
    Tensor xf = (x.dtype() == DType::F32) ? Tensor::empty(x.shape(), DType::F32, x.device())
                                           : cast(x, DType::F32);
    if (x.dtype() == DType::F32) {
        cudaMemcpy(xf.data_ptr(), x.data_ptr(), x.nbytes(), cudaMemcpyDeviceToDevice);
    }

    const int rows = static_cast<int>(x.numel() / x.shape().back());
    const int cols = static_cast<int>(x.shape().back());
    
    // Launch with a fixed power-of-two (256) for safe block reduction
    k_softmax_f32<<<rows, 256>>>(xf.data_as<float>(), cols);
    return xf;
}

// ─────────────────────────────────────────────────────────────
//  attention — GQA, causal
// ─────────────────────────────────────────────────────────────

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal, int sliding_window) {
    const int seq_len  = static_cast<int>(q.shape()[0]);
    const int n_heads  = static_cast<int>(q.shape()[1]);
    const int head_dim = static_cast<int>(q.shape()[2]);
    const int kv_len   = static_cast<int>(k.shape()[0]);
    const int n_kv     = static_cast<int>(k.shape()[1]);
    const int group    = n_heads / n_kv;

    const auto& dev = q.device();
    dev.make_current();

    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    Tensor out = Tensor::zeros({static_cast<std::size_t>(seq_len),
                                static_cast<std::size_t>(n_heads),
                                static_cast<std::size_t>(head_dim)},
                               DType::F32, dev);

    Tensor qf = cast(q, DType::F32);
    Tensor kf = cast(k, DType::F32);
    Tensor vf = cast(v, DType::F32);

    cublasHandle_t h = get_cublas(dev.index());

    for (int kv_h = 0; kv_h < n_kv; ++kv_h) {
        for (int g = 0; g < group; ++g) {
            int q_head = kv_h * group + g;

            const float* q_ptr = qf.data_as<float>() + q_head  * head_dim;
            const float* k_ptr = kf.data_as<float>() + kv_h * head_dim;
            const float* v_ptr = vf.data_as<float>() + kv_h * head_dim;

            Tensor scores = Tensor::empty(
                {static_cast<std::size_t>(seq_len), static_cast<std::size_t>(kv_len)},
                DType::F32, dev);

            const float alpha     = scale;
            const float beta_zero = 0.0f;
            const float alpha_one = 1.0f;

            cublas_check(
                cublasSgemmStridedBatched(
                    h,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    kv_len, seq_len, head_dim,
                    &alpha,
                    k_ptr, n_kv * head_dim, head_dim,
                    q_ptr, n_heads * head_dim, head_dim,
                    &beta_zero,
                    scores.data_as<float>(), kv_len, kv_len * seq_len,
                    1
                ),
                "attention: scores sgemm"
            );

            // FIXED: Apply both Causal and Sliding Window masks
            if (causal && seq_len > 1) {
                std::vector<float> s_host(seq_len * kv_len);
                cudaMemcpy(s_host.data(), scores.data_ptr(),
                           scores.nbytes(), cudaMemcpyDeviceToHost);
                           
                int seq_offset = kv_len - seq_len;
                for (int i = 0; i < seq_len; ++i) {
                    for (int j = 0; j < kv_len; ++j) {
                        bool mask = false;
                        // Causal mask: query cannot look at future keys
                        if (j > seq_offset + i) mask = true;
                        // SWA mask: query cannot look back further than sliding_window
                        if (sliding_window > 0 && j <= seq_offset + i - sliding_window) mask = true;
                        
                        if (mask) s_host[i * kv_len + j] = -1e38f;
                    }
                }
                
                cudaMemcpy(scores.data_ptr(), s_host.data(),
                           scores.nbytes(), cudaMemcpyHostToDevice);
            }

            Tensor attn = softmax(scores);

            float* out_ptr = out.data_as<float>() + q_head * head_dim;

            cublas_check(
                cublasSgemmStridedBatched(
                    h,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    head_dim, seq_len, kv_len,
                    &alpha_one,
                    v_ptr, n_kv * head_dim, head_dim,
                    attn.data_as<float>(), kv_len, kv_len * seq_len,
                    &beta_zero,
                    out_ptr, n_heads * head_dim, head_dim,
                    1
                ),
                "attention: context sgemm"
            );
        }
    }

    return cast(out, q.dtype());
}

// ─────────────────────────────────────────────────────────────
//  reshape, transpose, cast, cat, slice
// ─────────────────────────────────────────────────────────────

Tensor reshape(const Tensor& x, const Shape& shape) {
    if (shape.numel() != x.numel()) {
        throw std::invalid_argument("reshape: numel mismatch");
    }
    Tensor out = Tensor::empty(shape, x.dtype(), x.device());
    cudaMemcpy(out.data_ptr(), x.data_ptr(), x.nbytes(), cudaMemcpyDeviceToDevice);
    return out;
}

Tensor transpose(const Tensor& x, std::size_t dim_a, std::size_t dim_b) {
    if (x.rank() != 2) {
        throw std::invalid_argument("transpose: only 2D supported currently");
    }
    const int rows = static_cast<int>(x.shape()[0]);
    const int cols = static_cast<int>(x.shape()[1]);

    x.device().make_current();
    Tensor out = Tensor::empty(
        {static_cast<std::size_t>(cols), static_cast<std::size_t>(rows)},
        x.dtype(), x.device());

    if (x.dtype() == DType::F32) {
        cublasHandle_t h = get_cublas(x.device().index());
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgeam(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rows, cols,
            &alpha, x.data_as<float>(), cols,
            &beta,  nullptr, rows,
            out.data_as<float>(), rows);
    } else {
        throw std::invalid_argument("transpose: unsupported dtype");
    }
    return out;
}

__global__ void k_cast_bf16_to_f32(const __nv_bfloat16* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void k_cast_f32_to_bf16(const float* src, __nv_bfloat16* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

Tensor cast(const Tensor& x, DType target) {
    if (x.dtype() == target) {
        Tensor out = Tensor::empty(x.shape(), target, x.device());
        cudaMemcpy(out.data_ptr(), x.data_ptr(), x.nbytes(), cudaMemcpyDeviceToDevice);
        return out;
    }

    x.device().make_current();
    Tensor out = Tensor::empty(x.shape(), target, x.device());
    const int n = static_cast<int>(x.numel());
    const int blocks = (n + 255) / 256;

    if (x.dtype() == DType::BF16 && target == DType::F32) {
        k_cast_bf16_to_f32<<<blocks, 256>>>(x.data_as<__nv_bfloat16>(), out.data_as<float>(), n);
    } else if (x.dtype() == DType::F32 && target == DType::BF16) {
        k_cast_f32_to_bf16<<<blocks, 256>>>(x.data_as<float>(), out.data_as<__nv_bfloat16>(), n);
    } else {
        throw std::invalid_argument("cast: unsupported dtype pair");
    }
    return out;
}

Tensor cat(const std::vector<const Tensor*>& tensors, std::size_t dim) {
    if (tensors.empty()) throw std::invalid_argument("cat: empty tensor list");

    Shape out_shape = tensors[0]->shape();
    std::size_t cat_dim_size = out_shape[dim];
    for (std::size_t i = 1; i < tensors.size(); ++i) {
        cat_dim_size += tensors[i]->shape()[dim];
    }
    out_shape[dim] = cat_dim_size;

    const auto& dev = tensors[0]->device();
    dev.make_current();
    Tensor out = Tensor::empty(out_shape, tensors[0]->dtype(), dev);

    if (dim == 0) {
        std::size_t byte_offset = 0;
        for (const auto* t : tensors) {
            cudaMemcpy(
                static_cast<uint8_t*>(out.data_ptr()) + byte_offset,
                t->data_ptr(), t->nbytes(), cudaMemcpyDeviceToDevice
            );
            byte_offset += t->nbytes();
        }
    } else {
        throw std::invalid_argument("cat: only dim=0 supported currently");
    }

    return out;
}

Tensor slice(const Tensor& x, std::size_t dim, std::size_t begin, std::size_t end) {
    if (dim != 0) throw std::invalid_argument("slice: only dim=0 supported currently");

    Shape out_shape = x.shape();
    out_shape[0]    = end - begin;

    const std::size_t row_bytes = x.nbytes() / x.shape()[0];
    const auto& dev = x.device();
    dev.make_current();
    Tensor out = Tensor::empty(out_shape, x.dtype(), dev);
    cudaMemcpy(
        out.data_ptr(),
        static_cast<const uint8_t*>(x.data_ptr()) + begin * row_bytes,
        out.nbytes(),
        cudaMemcpyDeviceToDevice
    );
    return out;
}

// ─────────────────────────────────────────────────────────────
//  add_bias — broadcast 1D bias over 2D/3D tensor
// ─────────────────────────────────────────────────────────────

__global__ void k_add_bias_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int hidden
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = __bfloat162float(x[row * hidden + i]);
        float b = __bfloat162float(bias[i]);
        out[row * hidden + i] = __float2bfloat16(v + b);
    }
}

Tensor add_bias(const Tensor& x, const Tensor& bias) {
    const int rows   = static_cast<int>(x.numel() / x.shape().back());
    const int hidden = static_cast<int>(x.shape().back());

    x.device().make_current();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());

    const int threads = std::min(hidden, 256);
    k_add_bias_bf16<<<rows, threads>>>(
        x.data_as<__nv_bfloat16>(),
        bias.data_as<__nv_bfloat16>(),
        out.data_as<__nv_bfloat16>(),
        hidden
    );
    return out;
}

// ─────────────────────────────────────────────────────────────
//  Gemma-specific primitives (GELU, RMSNorm Offset, Scalar Mul)
// ─────────────────────────────────────────────────────────────

__global__ void k_gelu_bf16(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    // Tanh approximation of GELU
    const float c1 = 0.7978845608f; // sqrt(2/pi)
    const float c2 = 0.044715f;
    float val = 0.5f * v * (1.0f + tanhf(c1 * (v + c2 * v * v * v)));
    out[i]  = __float2bfloat16(val);
}

Tensor gelu(const Tensor& x) {
    x.device().make_current();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    const int n = static_cast<int>(x.numel());
    k_gelu_bf16<<<(n + 255) / 256, 256>>>(x.data_as<__nv_bfloat16>(), out.data_as<__nv_bfloat16>(), n);
    return out;
}

__global__ void k_rms_norm_offset_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int hidden, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float ss = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = __bfloat162float(x[row * hidden + i]);
        ss += v * v;
    }

    __shared__ float smem[256];
    smem[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = __bfloat162float(x[row * hidden + i]);
        // Gemma offset: add 1.0f to the weight
        float weight = __bfloat162float(w[i]) + 1.0f; 
        out[row * hidden + i] = __float2bfloat16(v * scale * weight);
    }
}

Tensor rms_norm_offset(const Tensor& x, const Tensor& weight, float eps) {
    const int seq_len = static_cast<int>(x.shape()[0]);
    const int hidden  = static_cast<int>(x.shape()[1]);

    x.device().make_current();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());

    const int threads = 256;
    k_rms_norm_offset_bf16<<<seq_len, threads>>>(
        x.data_as<__nv_bfloat16>(),
        weight.data_as<__nv_bfloat16>(),
        out.data_as<__nv_bfloat16>(),
        hidden, eps
    );
    return out;
}

__global__ void k_mul_scalar_bf16(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(__bfloat162float(x[i]) * scalar);
}

Tensor mul(const Tensor& x, float scalar) {
    x.device().make_current();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    const int n = static_cast<int>(x.numel());
    k_mul_scalar_bf16<<<(n + 255) / 256, 256>>>(x.data_as<__nv_bfloat16>(), out.data_as<__nv_bfloat16>(), scalar, n);
    return out;
}

} // namespace transformers::backend::ops