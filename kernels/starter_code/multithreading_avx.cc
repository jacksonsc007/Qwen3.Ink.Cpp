#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"
#include "immintrin.h"

struct multithreading_thread_args {
    int start, end;
    const struct matmul_params* params;
};

struct qwen_multithreading_thread_args {
    int start, end;
    const struct qwen_matmul_params* params;
};



static void * qwen_int4_multi_thread_avx_func(void * args_)
{
    qwen_multithreading_thread_args * args = (qwen_multithreading_thread_args *)args_;
    int start_col = args -> start;
    int end_col = args -> end;
    const qwen_matmul_params * params = args -> params;
    
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

    const int quantization_block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;
    // printf("\e[31m[INFO]\e[m avx int4 qwen\n");

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, quantization_block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    // printf("\e[31m[INFO]\e[m reference matrixmul begins\n");
    for (int row = 0; row < m; row++) {
        for (int col = start_col; col < end_col; col++) {
            __m256 acc0 = _mm256_setzero_ps();
            // Compute each block
            for (int ch = 0; ch < k;) 
            {
                // pointer to the start address of the int4 weights block.
                uint8_t *w_int4_ptr = &B->int4_data_ptr[(col * k + ch) / 2];
                // pointer to the start of the int8 activation block.
                const signed char *a_int8_ptr = &A->int8_data_ptr[row * k + ch];
                
                // scale of activation
                float s_a_1st     = params->A_scales[(row * k + ch) / quantization_block_size];
                float s_a_2nd     = params->A_scales[(row * k + ch) / quantization_block_size + 1];  // scales of the second block
                // scale of weight
                float s_w_1st     = params->scales[(col * k + ch) / quantization_block_size];
                float s_w_2nd     = params->scales[(col * k + ch) / quantization_block_size + 1];  // scales of the second block
                //  zero point of weight
                int8_t zero_w_1st = params->zero_point[(col * k + ch) / quantization_block_size];
                int8_t zero_w_2nd = params->zero_point[(col * k + ch) / quantization_block_size + 1];
                
                // 256 bits = 32 * 8 bit = 64 * 4 bit; Each __m256i holds 64 (2 blocks)int4 weights
                __m256i int4_weights_packed = _mm256_loadu_si256( (__m256i *) (w_int4_ptr));
                // get the first block and second block of weights
                // Each weight is stored as int8 for computation (TODOink: can we do int4 computation? Overflow issue?) 
                const __m256i lowMask = _mm256_set1_epi8(0x0F);
                __m256i int8_w_1st_blk =_mm256_and_si256(int4_weights_packed, lowMask); // 256 = 32 * 8 bit
                __m256i int8_w_2nd_blk =_mm256_and_si256(
                    _mm256_srli_epi16(int4_weights_packed, 4),
                    lowMask
                );
                // get the zero points for each block
                __m256i zp_1st_blk = _mm256_set1_epi8(zero_w_1st);
                __m256i zp_2nd_blk = _mm256_set1_epi8(zero_w_2nd);
                // apply zero point to the weights
                int8_w_1st_blk = _mm256_sub_epi8(int8_w_1st_blk, zp_1st_blk);
                int8_w_2nd_blk = _mm256_sub_epi8(int8_w_2nd_blk, zp_2nd_blk);

                // get the first block and second block of activation (int8)
                // 256bits = 32 * 8 bit. Thus we need to read twice.
                __m256i int8_a_1st_blk = _mm256_loadu_si256( (__m256i *) a_int8_ptr);
                __m256i int8_a_2nd_blk = _mm256_loadu_si256( (__m256i *) (a_int8_ptr + quantization_block_size));
                
                // _mm256_maddubs_epi16 only accept unsigned integer as its first arg, we need to transmit the sign bit
                // of weight to activation and use the tha absolute value of weight.
                // 1. get the abs weight
                __m256i abs_int8_w_1st_blk = _mm256_sign_epi8(int8_w_1st_blk, int8_w_1st_blk);
                __m256i abs_int8_w_2nd_blk = _mm256_sign_epi8(int8_w_2nd_blk, int8_w_2nd_blk);
                // 2. negate activation when weight is negative
                __m256i neg_int8_a_1st_blk = _mm256_sign_epi8(int8_a_1st_blk, int8_w_1st_blk);
                __m256i neg_int8_a_2nd_blk = _mm256_sign_epi8(int8_a_2nd_blk, int8_w_2nd_blk);
                // 3. int8 multiplication
                __m256i res_1st_blk = _mm256_maddubs_epi16(abs_int8_w_1st_blk, neg_int8_a_1st_blk);
                __m256i res_2nd_blk = _mm256_maddubs_epi16(abs_int8_w_2nd_blk, neg_int8_a_2nd_blk);
                
                // by now, the integer mul results are stored as 16-bit chunks. Before we
                // multiply them by 32-bit float scaling factors, we need to expand the result
                // to 32-bit and then convert then to 32-bit float.
                __m256i ones = _mm256_set1_epi16(0x01);
                res_1st_blk = _mm256_madd_epi16(ones, res_1st_blk); // 8 * 32 bit
                res_2nd_blk = _mm256_madd_epi16(ones, res_2nd_blk); // 8 * 32 bit
                // convert int32 to float
                __m256 res_1st_blk_fp = _mm256_cvtepi32_ps(res_1st_blk);
                __m256 res_2nd_blk_fp = _mm256_cvtepi32_ps(res_2nd_blk);
                
                // get the scaling factor
                // sp: scaling factor packed
                // __m256 sp_w_1st_blk = _mm256_set1_ps(s_w_1st);
                // __m256 sp_w_2nd_blk = _mm256_set1_ps(s_w_2nd);
                // __m256 sp_a_1st_blk = _mm256_set1_ps(s_a_1st);
                // __m256 sp_a_2nd_blk = _mm256_set1_ps(s_a_2nd);
                // fsp: fused scaling factor packed
                __m256 fsp_1st_blk = _mm256_set1_ps(s_w_1st * s_a_1st);
                __m256 fsp_2nd_blk = _mm256_set1_ps(s_w_2nd * s_a_2nd);
                
                // apply the scaling factors
                // TODOink: what if we use 2 accumulators?
                acc0 = _mm256_fmadd_ps(res_1st_blk_fp, fsp_1st_blk, acc0);
                acc0 = _mm256_fmadd_ps(res_2nd_blk_fp, fsp_2nd_blk, acc0);
                
                ch += quantization_block_size * 2;
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];

        }
    }
    return NULL;
}



namespace matmul {

void MatmulOperator::matMul_int4_multiThread_avx_qwen(struct qwen_matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32
    float *scale = params->scales, *offset = params->offset;
    // printf("\e[31m[INFO]\e[m mt qwen int4\n");
    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    const int n_threads = 8;
    const int cols_per_thread = (n / n_threads);

    pthread_t threads[n_threads];
    qwen_multithreading_thread_args qwen_th_args[n_threads];
    for (int i = 0; i < n_threads; i++)
    {
        qwen_th_args[i].start = (i)     * cols_per_thread;
        qwen_th_args[i].end   = (i + 1) * cols_per_thread;
        qwen_th_args[i].params = params;
        pthread_create(&threads[i], NULL, qwen_int4_multi_thread_avx_func, &qwen_th_args[i]);
    }
    for (int i = 0; i < n_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    // printf("\e[31m[INFO]\e[m reference matrixmul ends\n");
}


}  // namespace matmul
