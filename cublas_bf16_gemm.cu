#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;    \
      return 1;                                                                  \
    }                                                                            \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                          \
    cublasStatus_t status = (call);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS error: " << static_cast<int>(status) << std::endl;    \
      return 1;                                                                  \
    }                                                                            \
  } while (0)

__device__ __forceinline__ uint32_t mix32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__global__ void initialize(__nv_bfloat16* data, uint64_t count, uint32_t seed,
                           bool ones) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (; index < count; index += stride) {
    float value = 1.0f;
    if (!ones) {
      uint32_t bits = mix32(static_cast<uint32_t>(index) ^ seed);
      value = (static_cast<int>(bits & 0xffffU) - 32768) / 32768.0f;
    }
    data[index] = __float2bfloat16(value);
  }
}

__global__ void reference_c00(const __nv_bfloat16* a, const __nv_bfloat16* b,
                              float* result, int n) {
  __shared__ float partial[256];
  float sum = 0.0f;
  for (int k = threadIdx.x; k < n; k += blockDim.x) {
    sum += __bfloat162float(a[static_cast<uint64_t>(k) * n]) *
           __bfloat162float(b[k]);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset; offset >>= 1) {
    if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) *result = partial[0];
}

int main(int argc, char** argv) {
  int n = 16384;
  int warmup = 100;
  int iterations = 200;
  int repeats = 5;
  int interval_ms = 0;
  int sample_warmup = 0;
  std::string sample_warmup_mode = "same";
  std::string mode = "random";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--n" && ++i < argc) n = std::atoi(argv[i]);
    else if (arg == "--warmup" && ++i < argc) warmup = std::atoi(argv[i]);
    else if (arg == "--iterations" && ++i < argc) iterations = std::atoi(argv[i]);
    else if (arg == "--repeats" && ++i < argc) repeats = std::atoi(argv[i]);
    else if (arg == "--interval-ms" && ++i < argc) interval_ms = std::atoi(argv[i]);
    else if (arg == "--sample-warmup" && ++i < argc) sample_warmup = std::atoi(argv[i]);
    else if (arg == "--sample-warmup-mode" && ++i < argc) sample_warmup_mode = argv[i];
    else if (arg == "--mode" && ++i < argc) mode = argv[i];
    else {
      std::cerr << "invalid argument: " << arg << std::endl;
      return 2;
    }
  }
  if (n <= 0 || warmup < 0 || iterations <= 0 || repeats <= 0 || interval_ms < 0 ||
      sample_warmup < 0 ||
      (sample_warmup_mode != "same" && sample_warmup_mode != "one") ||
      (mode != "random" && mode != "one")) {
    std::cerr << "invalid benchmark parameters" << std::endl;
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(0));
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  int cublas_version = 0;
  CUBLAS_CHECK(cublasGetVersion(handle, &cublas_version));

  uint64_t elements = static_cast<uint64_t>(n) * n;
  size_t bytes = elements * sizeof(__nv_bfloat16);
  __nv_bfloat16 *a = nullptr, *b = nullptr, *c = nullptr;
  __nv_bfloat16 *warm_a = nullptr, *warm_b = nullptr;
  float* device_reference = nullptr;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMalloc(&c, bytes));
  CUDA_CHECK(cudaMalloc(&device_reference, sizeof(float)));
  int blocks = static_cast<int>(std::min<uint64_t>((elements + 255) / 256, 65535));
  bool ones = mode == "one";
  initialize<<<blocks, 256>>>(a, elements, 0x12345678U, ones);
  initialize<<<blocks, 256>>>(b, elements, 0x9abcdef0U, ones);
  if (sample_warmup > 0 && sample_warmup_mode == "one" && !ones) {
    CUDA_CHECK(cudaMalloc(&warm_a, bytes));
    CUDA_CHECK(cudaMalloc(&warm_b, bytes));
    initialize<<<blocks, 256>>>(warm_a, elements, 0U, true);
    initialize<<<blocks, 256>>>(warm_b, elements, 0U, true);
  }
  CUDA_CHECK(cudaGetLastError());

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto gemm_inputs = [&](const __nv_bfloat16* lhs, const __nv_bfloat16* rhs) {
    return cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, lhs, CUDA_R_16BF, n,
        rhs, CUDA_R_16BF, n, &beta, c, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  };
  auto gemm = [&]() { return gemm_inputs(a, b); };
  auto sample_warmup_gemm = [&]() {
    return warm_a ? gemm_inputs(warm_a, warm_b) : gemm();
  };

  CUBLAS_CHECK(gemm());
  reference_c00<<<1, 256>>>(a, b, device_reference, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  __nv_bfloat16 c00_bf16;
  float reference = 0.0f;
  CUDA_CHECK(cudaMemcpy(&c00_bf16, c, sizeof(c00_bf16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&reference, device_reference, sizeof(reference),
                        cudaMemcpyDeviceToHost));
  float c00 = __bfloat162float(c00_bf16);
  float relative_error = std::abs(c00 - reference) / std::max(1.0f, std::abs(reference));
  if (!std::isfinite(c00) || !std::isfinite(reference) || relative_error > 0.02f) {
    std::cerr << "validation failed: c00=" << c00 << " reference=" << reference
              << " relative_error=" << relative_error << std::endl;
    return 3;
  }

  for (int i = 0; i < warmup; ++i) CUBLAS_CHECK(gemm());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "CUTEX_BENCH_START" << std::endl;
  std::vector<double> samples_ms;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    if (interval_ms)
      std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    for (int i = 0; i < sample_warmup; ++i) CUBLAS_CHECK(sample_warmup_gemm());
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) CUBLAS_CHECK(gemm());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    samples_ms.push_back(elapsed_ms / iterations);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
  }
  std::cout << "CUTEX_BENCH_END" << std::endl;

  std::vector<double> ordered = samples_ms;
  std::sort(ordered.begin(), ordered.end());
  double median_ms = ordered[ordered.size() / 2];
  if (ordered.size() % 2 == 0)
    median_ms = 0.5 * (ordered[ordered.size() / 2 - 1] + ordered[ordered.size() / 2]);
  double mean_ms = std::accumulate(ordered.begin(), ordered.end(), 0.0) / ordered.size();
  double flops = 2.0 * n * n * n;
  double tflops = flops / median_ms / 1.0e9;

  std::cout << std::fixed << std::setprecision(9)
            << "{\"status\":\"PASS\",\"mode\":\"" << mode << "\",\"n\":" << n
            << ",\"warmup\":" << warmup << ",\"iterations\":" << iterations
            << ",\"repeats\":" << repeats << ",\"interval_ms\":" << interval_ms
            << ",\"sample_warmup\":" << sample_warmup
            << ",\"sample_warmup_mode\":\"" << sample_warmup_mode << "\""
            << ",\"flop_count\":"
            << static_cast<uint64_t>(2ULL * n * n * n)
            << ",\"median_ms\":" << median_ms << ",\"mean_ms\":" << mean_ms
            << ",\"min_ms\":" << ordered.front() << ",\"max_ms\":" << ordered.back()
            << ",\"tflops\":" << tflops << ",\"c00\":" << c00
            << ",\"reference_c00\":" << reference
            << ",\"relative_error_c00\":" << relative_error
            << ",\"cublas_version\":" << cublas_version
            << ",\"repeat_average_ms\":[";
  for (size_t i = 0; i < samples_ms.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << samples_ms[i];
  }
  std::cout << "]}" << std::endl;

  CUDA_CHECK(cudaFree(device_reference));
  if (warm_b) CUDA_CHECK(cudaFree(warm_b));
  if (warm_a) CUDA_CHECK(cudaFree(warm_a));
  CUDA_CHECK(cudaFree(c));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(a));
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
