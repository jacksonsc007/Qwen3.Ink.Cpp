#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <immintrin.h>

#include "common.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"
#include "QwenOperator.h"
// #include <sstream>

// Standalone repack function for testing
void repack_w81_weight_test(const int K, const int N, const int Q_BLK_SIZE, void * B_repack, 
                           const float * SB, const float * MinB, const uint8_t * B) {
    // Simplified implementation for testing - just copy data
    // In a real implementation, this would do the actual repacking
    int num_repack_blk_B_along_K = K / (2 * Q_BLK_SIZE);
    int num_repack_blk_B_along_N = N / 8;
    struct q4_repack_2x8_fp16 * B_start = (struct q4_repack_2x8_fp16 *) B_repack;
    
    for (int j = 0; j < num_repack_blk_B_along_N; j++) {
        for (int i = 0; i < num_repack_blk_B_along_K; i++) {
            struct q4_repack_2x8_fp16 * B_ptr = B_start + j * num_repack_blk_B_along_K + i;
            
            // Pack scaling factors
            for (int jj = 0; jj < 8; jj++) {
                fp16_t s_low = GGML_FP32_TO_FP16(SB[i * 2 + 0 * (K / Q_BLK_SIZE) + j * 8 + jj]);
                fp16_t s_high = GGML_FP32_TO_FP16(SB[i * 2 + 1 * (K / Q_BLK_SIZE) + j * 8 + jj]);
                fp16_t min_low = GGML_FP32_TO_FP16(MinB[i * 2 + 0 * (K / Q_BLK_SIZE) + j * 8 + jj]);
                fp16_t min_high = GGML_FP32_TO_FP16(MinB[i * 2 + 1 * (K / Q_BLK_SIZE) + j * 8 + jj]);
                
                B_ptr->s_low[jj] = s_low;
                B_ptr->s_high[jj] = s_high;
                B_ptr->min_low[jj] = min_low;
                B_ptr->min_high[jj] = min_high;
            }
            
            // Pack quantized weights (simplified)
            uint8_t * B_ptr_start = B_ptr->q_coupled;
            int num_packs = 8;
            for (int pack_idx = 0; pack_idx < num_packs; pack_idx++) {
                uint8_t * pack_start_ptr = B_ptr_start + pack_idx * 32;
                for (int jj = 0; jj < 8; jj++) {
                    for (int ii = 0; ii < 4; ii++) {
                        pack_start_ptr[ii + jj * 4] = B[i * Q_BLK_SIZE + pack_idx * 4 + ii + j * 8 + jj * (K / 2)];
                    }
                }
            }
        }
    }
}

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
    size_t size_per_q_block = (
        sizeof(float) * 4 + // scaling factors, min, zero, scaled sum
        (sizeof(uint8_t) / 2) * QK + // quantized weights
        (sizeof(uint8_t) / 2) * QK // prepare for mixture of A80W40 and A81W41
    );
    size_t weight_buffer_size = (
        num_q_blocks * size_per_q_block
    );
    ctx.repack_buffer = std::make_unique<int8_t []>(weight_buffer_size); // 2 int4 weights in one byte

    size_t max_activation_cols = 12288; // 12288 is the maximum hidden dimension
    size_t max_activation_rows = 4096;
    num_q_blocks = max_activation_rows * max_activation_cols / QK;
    size_per_q_block = (
        sizeof(float) * 4 + // scaling factors, min, zero, scaled sum
        (sizeof(uint8_t)) * QK 
    );

    size_t activation_buffer_size = (
        num_q_blocks * size_per_q_block
    );
    ctx.activation_buffer = std::make_unique<int8_t []>(activation_buffer_size); // 2 int4 weights in one byte

    
    //
    Qwen_Linear_with_bias_Int4 q_proj(&ctx, param_path + "/q_proj/", 1, q_dim, hidden_dim);
    
    Matrix3D<float> input(1, sqlen, hidden_dim);
    // Matrix3D<float> output_gt(1, sqlen, hidden_dim);
    std::string input_path = "tests/assets/qwen3_attention_input.bin";
    std::string output_gt_path = "tests/assets/qwen3_attention_q_unshape.bin";
    input.load(input_path.c_str());
    // output_gt.load(output_gt_path.c_str());
    
    Matrix3D<float> output_linear = q_proj.forward_gemv(input);
    output_linear.compare_with_gt(output_gt_path);
}

void test_matmul_kernel_throughput(int m, int n, int k) {
// void test_linear_implementation_throughput() {
    const int MEM_ALIGN = 64;
    // const int m = 1024;
    // const int n = 1024;
    // const int k = 1024;
    
    
    // Allocate memory using _mm_malloc with 64-byte alignment
    float* activation_data = static_cast<float*>(_mm_malloc(m * k * sizeof(float), MEM_ALIGN));
    float* weight_data = static_cast<float*>(_mm_malloc(n * k * sizeof(float), MEM_ALIGN));
    float* output_data = static_cast<float*>(_mm_malloc(m * n * sizeof(float), MEM_ALIGN));
    float* activation_repack_data = static_cast<float*>(_mm_malloc(m * k * sizeof(float), MEM_ALIGN));
    float* weight_repack_data = static_cast<float*>(_mm_malloc(n * k * sizeof(float), MEM_ALIGN));

    // Initialize test data with random values
    for (int i = 0; i < m * k; i++) {
        activation_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < n * k; i++) {
        weight_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Create quantization parameter matrices
    float* scale_data = static_cast<float*>(_mm_malloc(n * (k / 32) * sizeof(float), MEM_ALIGN));  // QK = 32, so k / 32 blocks
    float* offset_data = static_cast<float*>(_mm_malloc(n * (k / 32) * sizeof(float), MEM_ALIGN));
    uint8_t* q4_w_data = static_cast<uint8_t*>(_mm_malloc(n * (k / 2) * sizeof(uint8_t), MEM_ALIGN));  // Each uint8_t contains two int4 weights

    // Initialize quantization parameters
    for (int i = 0; i < n * (k / 32); i++) {
        scale_data[i] = 1.0f;  // Default scale
        offset_data[i] = 2.0f; // Default offset
    }
    
    // Initialize quantized weights (simplified - in practice this would be quantized)
    for (int i = 0; i < n * (k / 2); i++) {
        q4_w_data[i] = static_cast<uint8_t>(rand() % 256);
    }

    int8_t * A_repack = reinterpret_cast<int8_t *>(activation_repack_data);
    quantize_row_q8_1_repack_fp16(activation_data, A_repack, m * k);
    repack_w81_weight_test(k, n, 32, weight_repack_data, scale_data, offset_data, q4_w_data);

    // Measure throughput
    const int num_iterations = 100;
    const long long ops = 2 * (long long)m * (long long)n * (long long)k;
    std::ostringstream oss;
    oss << "[" << "test_linear" << ": " << m << " x " << n << " x " << k << "]";
    std::string formatted_profile_name = oss.str();
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        gemm_repack_A81W41_fp32(
            A_repack, weight_repack_data, output_data,
            m, n, k
        );
    }
    
    // Measure throughput
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i)
    {
        PROFILE_START_FLOPS(formatted_profile_name, ops);
         gemm_repack_A81W41_fp32(
            A_repack, weight_repack_data, output_data,
            m, n, k
        );
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
    printf("  Total operations: %lld\n", total_ops);
    printf("  Throughput: %.2f GFLOPS\n", throughput_gflops);
    printf("  Matrix dimensions: %d x %d x %d\n", m, n, k);
    
    // Free allocated memory
    _mm_free(activation_data);
    _mm_free(weight_data);
    _mm_free(output_data);
    _mm_free(activation_repack_data);
    _mm_free(weight_repack_data);
    _mm_free(scale_data);
    _mm_free(offset_data);
    _mm_free(q4_w_data);
}

int main() {
    // test_linear_implementation_correctness();
    // test_linear_implementation_throughput();
    test_matmul_kernel_throughput(1024, 1024, 1024);
    test_matmul_kernel_throughput(320, 12288, 4096);
    // test_linear_implementation_throughput(320, 1024, 4096);
    Profiler::getInstance().report_internal();
}
