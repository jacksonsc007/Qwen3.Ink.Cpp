#include <stdint.h>
#include <sys/time.h>
#include <stdio.h>

void quantize_fp32_to_int8(float *A, int8_t *qA, float *sA, int size, int block_size);
void quantize_fp32_to_int8_q80(float *A, int8_t *qA, float *sA, int size, int block_size);
void quantize_fp32_to_int8_q81(float* A, int8_t* qA, float* sA, float* scaledSumA, int size, int block_size);

// Data structures
struct quantization_params {
    float scale;
    bool per_channel = false;
    int32_t zero_point;
    int8_t q_min = -128, q_max = 127;
};

struct matrix {
    int row;
    int column;
    float *data_ptr;
    int32_t *int32_data_ptr;
    int8_t *int8_data_ptr;
    uint8_t *uint8_data_ptr;
    uint8_t *int4_data_ptr;
    struct quantization_params qparams;
    size_t length() const { return row * column; }
};

struct optimization_params {
    int blk_size; // the size of cache block
    int num_thread = 8;
};

struct qwen_matmul_params {
    struct matrix A, B, C, bias;
    struct optimization_params opt_params;
    float alpha, beta;
    // for int4
    float *scales, *offset;
    int8_t * zero_point;
    int block_size; // the size of quantization block
    // for int8 activation
    float *A_scales;
    int8_t* A_zero_point;
    float *A_scaled_sum;
};

struct matmul_params {
    struct matrix A, B, C, bias;
    struct optimization_params opt_params;
    float alpha, beta;
    // for int4
    float *scales, *offset;
    float * zero_point;
    int block_size; // the size of quantization block
    // for int8 activation
    float *A_scales;
    int8_t A_zero_point;
    float *A_scaled_sum;
};

struct thread_args {
    const struct matrix *A;
    const struct matrix *B;
    const struct matrix *C;
    const struct matmul_params *params;
    int start_i, end_i, blk_size;
};

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))
namespace matmul {
class MatmulOperator {
   public:
    void mat_mul_transposed(const struct matmul_params *params);
    void mat_mul_accelerator_transposed_fastover_column(const struct matmul_params *params);
    // int8
    void naive_mat_mul_int8(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(const struct matmul_params *params);
    // int4
    void mat_mul_accelerator_int4_fast(const struct matmul_params *params);
    void mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params *params);
    void mat_mul_accelerator_int8_int4_fast_no_offset(struct matmul_params *params);
    void naive_mat_mul_int4(const struct matmul_params *params);
    void naive_mat_mul_int4_with_offset(const struct matmul_params *params);
    // fp32
    void mat_mul_loop_unrolling4x1_fp32(struct matmul_params *params);
    void mat_mul_loop_unrolling4x4_fp32(struct matmul_params *params);
    void mat_mul_loop_unrolling8x8_fp32(struct matmul_params *params);
    void mat_mul_accelerator_transposed_fastover_column_fp32_avx(const struct matmul_params *params);
    void mat_mul_loop_unrolling4x4_mt_fp32(struct matmul_params *params);
    void mat_mul_loop_unrolling4x4_mt_avx_fp32(struct matmul_params *params);
    void mat_mul_loop_unrolling4x4_avx_fp32(struct matmul_params *params);
    void mat_mul_avx_fp32(struct matmul_params *params);
    void mat_mul_tiling_fp32(struct matmul_params *params);
    void mat_mul_loop_unrolling4x4_tiling_fp32(struct matmul_params *params);
    void mat_mul_multithreading_fp32(struct matmul_params* params);
    void mat_mul_multithreading_tiling_fp32(struct matmul_params* params);
    void mat_mul_loop_unrolling_second_innermost_4x4_fp32(struct matmul_params *params);
    void matMul_loopUnrolling2level4x4_fp32(struct matmul_params *params);
    void matMul_loopUnrollingSecondInnermost4x4_avx_fp32(struct matmul_params *params) ;
    void mat_mul_loop_unrolling4x4_secondInnermost_tiling_fp32(struct matmul_params *params);
    void matMul_tiling2level_fp32(struct matmul_params *params);
    void mat_mul_loop_unrolling4x4_secondInnermost_tiling2level_fp32(struct matmul_params *params);
    void matMul_avx_tiling_fp32(struct matmul_params *params);
    void matMul_avx_tiling2level_fp32(struct matmul_params *params);
    
    // int4 qwen
    void matMul_int4Reference_pseudoQ_qwen(struct qwen_matmul_params *params);
    void matMul_int4Reference_qwen(struct qwen_matmul_params *params);
    void matMul_int4_Tiling1vl_qwen(struct qwen_matmul_params *params);
    void matMul_int4_unrolling2x2_qwen(struct qwen_matmul_params *params);
    void matMul_int4_avx_qwen(struct qwen_matmul_params *params);
    void matMul_int4_multiThread_qwen(struct qwen_matmul_params *params);
    void matMul_int4_multiThread_avx_qwen(struct qwen_matmul_params *params);
    void qgemv_A80W4z_kernel(struct qwen_matmul_params *params);
    void qgemm_A80W4z_kernel(struct qwen_matmul_params *params);
    void qgemv_A80W40_kernel(struct qwen_matmul_params *params);
    void qgemm_A80W40_kernel(struct qwen_matmul_params *params);



    // w8a4 code template functions
    void mat_mul_reference(struct matmul_params *params);
    void mat_mul_loop_unrolling(struct matmul_params *params);
    void mat_mul_multithreading(struct matmul_params *params);
    void mat_mul_multithreading_loop_unrolling(struct matmul_params *params);
    void mat_mul_simd_programming(struct matmul_params *params);
    void mat_mul_all_techniques(struct matmul_params *params);
    // cuda
    void mat_mul_cuda(const struct matmul_params *params);

   private:
    float interval_to_us(struct timeval *start, struct timeval *end);
    void CHECK_MATRICES(const struct matrix *A, const struct matrix *B, const struct matrix *C);
    void CHECK_MATRICES_int4weight(const struct matrix *A, const struct matrix *B, const struct matrix *C);
};
}  // namespace matmul
