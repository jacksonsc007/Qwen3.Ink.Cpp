#include <immintrin.h>
#include <mm_malloc.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "QwenOperator.h"
#include "common.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"
// #include <sstream>

void test_linear_implementation_correctness() {
    std::string param_path = "model_weights/int4/qwen3-8b-A81W41/model/layers/layer0/self_attn";
    // std::string param_path = "model_weights/int4/qwen3_8b_A80W40/model/layers/layer0/self_attn";
    int q_dim = 4096;
    int hidden_dim = 4096;
    int sqlen = 10;
    ModelContext ctx;
    size_t max_weight_rows = 4096;
    size_t max_weight_cols = 151936;
    size_t num_q_blocks = max_weight_rows * max_weight_cols / QK;
    size_t size_per_q_block = (sizeof(float) * 4 +           // scaling factors, min, zero, scaled sum
                               (sizeof(uint8_t) / 2) * QK +  // quantized weights
                               (sizeof(uint8_t) / 2) * QK    // prepare for mixture of A80W40 and A81W41
    );
    size_t weight_buffer_size = (num_q_blocks * size_per_q_block);
    ctx.repack_buffer = make_aligned_x86<int8_t>(64, weight_buffer_size);

    size_t max_activation_cols = 12288;  // 12288 is the maximum hidden dimension
    size_t max_activation_rows = 4096;
    num_q_blocks = max_activation_rows * max_activation_cols / QK;
    size_per_q_block = (sizeof(float) * 4 +  // scaling factors, min, zero, scaled sum
                        (sizeof(uint8_t)) * QK);

    size_t activation_buffer_size = (num_q_blocks * size_per_q_block);
    ctx.activation_buffer = make_aligned_x86<int8_t>(64, activation_buffer_size);

    //
    Qwen_Linear_with_bias_Int4 q_proj(&ctx, param_path + "/q_proj/", 1, q_dim, hidden_dim);

    Matrix3D<float> input(1, sqlen, hidden_dim);
    // Matrix3D<float> output_gt(1, sqlen, hidden_dim);
    std::string input_path = "tests/assets/qwen3_attention_input.bin";
    std::string output_gt_path = "tests/assets/qwen3_attention_q_unshape.bin";
    input.load(input_path.c_str());
    // output_gt.load(output_gt_path.c_str());

    Matrix3D<float> output_linear = q_proj.forward(input);
    output_linear.compare_with_gt(output_gt_path);
}

void test_matmul_kernel_throughput(int m, int n, int k) {
    Matrix3D<float> activation(1, m, k);
    Matrix3D<float> weight(1, n, k);
    // Matrix3D<float> output(1, m, n);
    // Matrix3D<float> activation_repack(1, m, k);
    // Matrix3D<float> weight_repack(1, n, k);
    const int MEM_ALIGN = 64;
    float* activation_repack_data = static_cast<float*>(_mm_malloc(m * k * sizeof(float), MEM_ALIGN));
    float* weight_repack_data = static_cast<float*>(_mm_malloc(n * k * sizeof(float), MEM_ALIGN));
    float* output_data = static_cast<float*>(_mm_malloc(m * n * sizeof(float), MEM_ALIGN));

    // Initialize test data with random values
    for (int i = 0; i < m * k; i++) {
        activation.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < n * k; i++) {
        weight.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Create quantization parameter matrices
    Matrix3D<float> scale(1, n, k / 32);  // QK = 32, so k / 32 blocks
    Matrix3D<float> offset(1, n, k / 32);
    Matrix3D<uint8_t> q4_w(1, n, k / 2);  // Each uint8_t contains two int4 weights

    // Initialize quantization parameters
    for (int i = 0; i < n * (k / 32); i++) {
        scale.data()[i] = 1.0f;   // Default scale
        offset.data()[i] = 2.0f;  // Default offset
    }

    // Initialize quantized weights (simplified - in practice this would be quantized)
    for (int i = 0; i < n * (k / 2); i++) {
        q4_w.data()[i] = static_cast<uint8_t>(rand() % 256);
    }

    int8_t* A_repack = reinterpret_cast<int8_t*>(activation_repack_data);
    quantize_row_q8_1_repack_fp32(activation.data(), A_repack, m * k);
    repack_w81_weight_fp32(k, n, 32, weight_repack_data, scale.data(), offset.data(), q4_w.data());

    // Measure throughput
    const int num_iterations = 100;
    const long long ops = 2 * (long long)m * (long long)n * (long long)k;
    std::ostringstream oss;
    oss << "[" << "standalone matmul kernel" << ": " << m << " x " << n << " x " << k << "]";
    std::string formatted_profile_name = oss.str();

    // Warm up
    for (int i = 0; i < 10; ++i) {
        gemm_repack_A81W41_fp32(A_repack, weight_repack_data, output_data, m, n, k);
    }

    // Measure throughput
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        quantize_row_q8_1_repack_fp32(activation.data(), A_repack, m * k);
        PROFILE_START_FLOPS(formatted_profile_name, ops);
        gemm_repack_A81W41_fp32(A_repack, weight_repack_data, output_data, m, n, k);
        PROFILE_END(formatted_profile_name);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate throughput
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double total_time_seconds = duration.count() / 1000000.0;
    double avg_time_per_iteration = total_time_seconds / num_iterations;
    double total_ops = ops * num_iterations;
    double throughput_gflops = (total_ops / total_time_seconds) / 1e9;

    printf("Throughput measurement for %s:\n", formatted_profile_name.c_str());
    printf("  Total time: %.6f seconds\n", total_time_seconds);
    printf("  Average time per iteration: %.6f seconds\n", avg_time_per_iteration);
    printf("  Total operations: %f\n", total_ops);
    printf("  Throughput: %.2f GFLOPS\n", throughput_gflops);
    printf("  Matrix dimensions: %d x %d x %d\n", m, n, k);

    _mm_free(activation_repack_data);
    _mm_free(weight_repack_data);
    _mm_free(output_data);
}

void test_mlp_linear_layer_throughput() {
    std::string param_path = "model_weights/int4/qwen3-8b-A81W41/model/layers/layer0";
    ModelContext ctx;
    size_t max_weight_rows = 4096;
    size_t max_weight_cols = 151936;
    size_t num_q_blocks = max_weight_rows * max_weight_cols / QK;
    size_t size_per_q_block = (sizeof(float) * 4 +           // scaling factors, min, zero, scaled sum
                               (sizeof(uint8_t) / 2) * QK +  // quantized weights
                               (sizeof(uint8_t) / 2) * QK    // prepare for mixture of A80W40 and A81W41
    );
    size_t weight_buffer_size = (num_q_blocks * size_per_q_block);
    ctx.repack_buffer = make_aligned_x86<int8_t>(64, weight_buffer_size);

    size_t max_activation_cols = 12288;  // 12288 is the maximum hidden dimension
    size_t max_activation_rows = 4096;
    num_q_blocks = max_activation_rows * max_activation_cols / QK;
    size_per_q_block = (sizeof(float) * 4 +  // scaling factors, min, zero, scaled sum
                        (sizeof(uint8_t)) * QK);

    size_t activation_buffer_size = (num_q_blocks * size_per_q_block);
    ctx.activation_buffer = make_aligned_x86<int8_t>(64, activation_buffer_size);

    //
    int sqlen = 320;
    int mlp_proj_dim = 12288;
    int hidden_dim = 4096;
    Qwen_Linear_with_bias_Int4 q_proj(&ctx, param_path + "/mlp/up_proj/", 1, mlp_proj_dim, hidden_dim);
    Matrix3D<float> input(1, sqlen, hidden_dim);
    // init input
    for (int i = 0; i < input.length(); ++i) {
        input.data()[i] = 0.2;
    }

    // Matrix3D<float> output_gt(1, sqlen, hidden_dim);
    // std::string input_path = "tests/assets/qwen3_attention_input.bin";
    // std::string output_gt_path = "tests/assets/qwen3_attention_q_unshape.bin";
    // input.load(input_path.c_str());
    // output_gt.load(output_gt_path.c_str());

    const int num_iters = 20;
    for (int i = 0; i < num_iters; ++i) {
        Matrix3D<float> output_linear = q_proj.forward(input);
    }
}

int main() {
    // NOTE: we must call ggml_init before invoking GGML_FP16_TO_FP32
    const int ctx_size = 0;
    struct ggml_init_params params = {/*.mem_size   =*/ctx_size,
                                      /*.mem_buffer =*/NULL,
                                      /* no_alloc   =*/0};

    struct ggml_context* ctx;
    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return 1;
    }
    float a = 1.30;
    ggml_fp16_t a_fp16 = GGML_FP32_TO_FP16(a);
    float a_ = GGML_FP16_TO_FP32(a_fp16);
    printf("%f vs %f\n", a, a_);

    // test_linear_implementation_correctness();
    test_mlp_linear_layer_throughput();
    test_matmul_kernel_throughput(320, 12288, 4096);
    // test_matmul_kernel_throughput(320, 1024, 4096);
    Profiler::getInstance().report_internal();
}
