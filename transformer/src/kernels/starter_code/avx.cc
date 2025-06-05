#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

// include avx intrinsics
#include <immintrin.h>
#include <xmmintrin.h>
#include "matmul.h"

namespace matmul {

void MatmulOperator::matMul_int4_avx_qwen(struct qwen_matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

    const int quantization_block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;
    // printf("\e[31m[INFO]\e[m avx int4 qwen\n");

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, quantization_block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    // printf("\e[31m[INFO]\e[m reference matrixmul begins\n");
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
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
    // printf("\e[31m[INFO]\e[m reference matrixmul ends\n");
}


// void MatmulOperator::mat_mul_avx_fp32(struct matmul_params *params) {
//     const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
//     float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

//     assert(A->column == B->row);
//     assert(C->row == A->row);
//     assert(C->column == B->column);
//     int m = A->row, n = B->column, k = A->column;

//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             float acc = 0;
//             __m256 acc_packed = _mm256_setzero_ps();
//             int kk = 0;
//             for (; kk <= k-8; kk+=8) {
//                 __m256 * a_ptr = (__m256 *) (data_A + i * k + kk);
//                 __m256 * b_ptr = (__m256 *) (data_B + j * k + kk);
//                 __m256 a_val = _mm256_loadu_ps((float *) a_ptr);
//                 __m256 b_val = _mm256_loadu_ps((float *) b_ptr);
//                 acc_packed = _mm256_add_ps(acc_packed, _mm256_mul_ps(a_val, b_val));
//             }
            
//             for (;kk < k; kk++)
//             {
//                 acc += data_A[i * k + kk] * data_B[j * k + kk];                
//             }
//             float * ptr0 = (float*) &acc_packed;
//             acc  += ptr0[0] + ptr0[1] + ptr0[2] + ptr0[3] + 
//                         ptr0[4] + ptr0[5] + ptr0[6] + ptr0[7];
//             data_C[i * n + j] = acc;
//         }
//     }
// }

// void MatmulOperator::mat_mul_avx_fp32(struct matmul_params *params) {
//     const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
//     float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

//     assert(A->column == B->row);
//     assert(C->row == A->row);
//     assert(C->column == B->column);
//     int m = A->row, n = B->column, k = A->column;

//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             __m256 acc_packed = _mm256_setzero_ps();
//             int kk = 0;
//             for (; kk <= k-8; kk+=8) {
//                 __m256 a_val = _mm256_loadu_ps(data_A + i * k + kk);
//                 __m256 b_val = _mm256_loadu_ps(data_B + j * k + kk);
//                 acc_packed = _mm256_add_ps(acc_packed, _mm256_mul_ps(a_val, b_val));
//             }
//             float * ptr0 = (float*) &acc_packed;
//             data_C[i * n + j] = ptr0[0] + ptr0[1] + ptr0[2] + ptr0[3] + 
//                                 ptr0[4] + ptr0[5] + ptr0[6] + ptr0[7];

//         }
//     }
// }


void MatmulOperator::mat_mul_avx_fp32(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            __m256 acc_packed = _mm256_setzero_ps();
            int kk = 0;
            for (; kk <= k-8; kk+=8) {
                __m256 a_val = _mm256_loadu_ps(data_A + i * k + kk);
                __m256 b_val = _mm256_loadu_ps(data_B + j * k + kk);
                // acc_packed = _mm256_add_ps(acc_packed, _mm256_mul_ps(a_val, b_val));
                acc_packed  = _mm256_fmadd_ps(a_val, b_val, acc_packed);
            }
            float * ptr0 = (float*) &acc_packed;
            data_C[i * n + j] = ptr0[0] + ptr0[1] + ptr0[2] + ptr0[3] + 
                                ptr0[4] + ptr0[5] + ptr0[6] + ptr0[7];

        }
    }
}

}  // namespace matmul
