#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

// #define QM_x86

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#ifdef QM_x86
#include <immintrin.h>
#endif
namespace matmul {






#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "stdbool.h"

#include "lib.h"

#ifndef NTHREADS
#define NTHREADS 8
#endif
#define OMP_SCHEDULE dynamic
#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")

#define MEM_ALIGN 64
#define Q_BLK_SIZE 32
#define MR 4
#define NR 4
#define MC MR * 11 * NTHREADS
#define NC NR * 25 * NTHREADS
// #define MC (6 * (800 / NTHREADS) * NTHREADS)
// #define NC (16 * (40 / NTHREADS) * NTHREADS)
#define KC 1024 

#define min(a, b) ((a) < (b) ? (a) : (b))

// row-major order: matrix A and C, SA
// column-major order: matrix B, SB
// A: M x K B: K x N C: M x N
#define A(i, j, ld) ( A + ( i ) * ( ld ) + ( j ) )
#define B(i, j, ld) ( B + ( j ) * ( ld ) + ( i ) )
#define C(i, j, ld) ( C + ( i ) * ( ld ) + ( j ) )
#define SA(i, j, ld) (SA + (i) * (ld) + (j))
#define ScaledSumA(i, j, ld) (ScaledSumA + (i) * (ld) + (j))
#define SB(i, j, ld) (SB + (j) * (ld) + (i))
#define MinB(i, j, ld) (MinB + (j) * (ld) + (i))

// TODO: Design store pattern that is friendly to the access pattern of int8 weight and activation?

// blockA: packed. row-major.      (MR, KC)
// blockB: packed. column-major.   (KC, NR)
// blockSA: packed. row-major.     (MR, KC/Q_BLK_SIZE)
// blockSB: packed. column-major.  (KC/Q_BLK_SIZE, NR)
#define blockA(i, j, ld) ( blockA  + (i) * (ld) + (j) )
#define blockSA(i, j, ld) (blockSA + (i) * (ld) + (j))
#define blockScaledSumA(i, j, ld) (blockScaledSumA + (i) * (ld) + (j))
#define blockB(i, j, ld) ( blockB  + (j) * (ld) + (i) )
#define blockSB(i, j, ld) (blockSB + (j) * (ld) + (i))
#define blockMinB(i, j, ld) (blockMinB + (j) * (ld) + (i))


static int8_t blockA_packed[MC * KC] __attribute__((aligned(64)));
static float sA_packed[MC * KC / Q_BLK_SIZE] __attribute__((aligned(64)));
static float scaledSumA_packed[MC * KC / Q_BLK_SIZE] __attribute__((aligned(64)));
static uint8_t blockB_packed[KC / 2 * NC] __attribute__((aligned(64)));
static float sB_packed[NC * KC / Q_BLK_SIZE] __attribute__((aligned(64)));
static float minB_packed[NC * KC / Q_BLK_SIZE] __attribute__((aligned(64)));

/*
blockA: packed. row-major.      (MR, KC)
blockB: packed. column-major.   (KC, NR)
blockSA: packed. row-major.     (MR, KC/Q_BLK_SIZE)
blockSB: packed. column-major.  (KC/Q_BLK_SIZE, NR)
*/
void matmul_kernel(
    int8_t* A, uint8_t* B, float* C,
    float* SA, float* ScaledSumA, float* SB, float* MinB,
    const int M, const int N, const int K) 
{

    // Process 4 rows and 4 columns at a time
        PRAGMA_OMP_PARALLEL_FOR
        for (int j = 0; j < N; j += 4) 
        for (int i = 0; i < M; i += 4) 
        {
            
            // Accumulators for all combinations of i and j
            __m256 sum_packed_fp[4][4];
            float extra_sum[4][4];
            // float sum_a_fused[4][4];
            // float min_b[4][4];
            for (int ii = 0; ii < 4; ++ii) {
                for (int jj = 0; jj < 4; ++jj) {
                    sum_packed_fp[ii][jj] = _mm256_setzero_ps();
                    extra_sum[ii][jj] = 0;
                    // min_b[ii][jj] = 0.01;
                    // sum_a_fused[ii][jj] = 1115.6;
                }
            }

            for (int p = 0; p < K; p += 2 * Q_BLK_SIZE) 
            {
                int block_idx = p / Q_BLK_SIZE;
                
                // Load scaling factors for all active i and j
                float sA[4][2]; // [i][0/1 for low/high block]
                float sB[4][2]; // [j][0/1 for low/high block]
                
                for (int ii = 0; ii < 4; ++ii) {
                        sA[ii][0] = *SA(i+ii, block_idx, K/Q_BLK_SIZE);
                        sA[ii][1] = *SA(i+ii, block_idx+1, K/Q_BLK_SIZE);
                }
                
                for (int jj = 0; jj < 4; ++jj) {
                        sB[jj][0] = *SB(block_idx, j+jj, K/Q_BLK_SIZE);
                        sB[jj][1] = *SB(block_idx+1, j+jj, K/Q_BLK_SIZE);
                }
                
                // Load A blocks for active i
                __m256i a_pack_int8[4][2]; // [i][0/1 for low/high block]
                for (int ii = 0; ii < 4; ++ii) {
                        a_pack_int8[ii][0] = _mm256_loadu_si256((__m256i *) A(i+ii, p, K));
                        a_pack_int8[ii][1] = _mm256_loadu_si256((__m256i *) A(i+ii, p+Q_BLK_SIZE, K));
                }
                
                // Load B blocks for active j
                __m256i b_pack_int8[4]; // [j]
                __m256i b_pack_int8_low[4], b_pack_int8_high[4];
                for (int jj = 0; jj < 4; ++jj) {
                        b_pack_int8[jj] = _mm256_loadu_si256((__m256i *) B(p/2, j+jj, K/2));
                        
                        __m256i mask = _mm256_set1_epi8(0x0f);
                        b_pack_int8_low[jj] = _mm256_and_si256(b_pack_int8[jj], mask);
                        b_pack_int8_high[jj] = _mm256_and_si256(_mm256_srli_epi16(b_pack_int8[jj], 4), mask);
                }
                
                // Compute all active i/j combinations
                for (int ii = 0; ii < 4; ++ii) {
                    
                    for (int jj = 0; jj < 4; ++jj) {
                        
                        float fused_s[2] = {
                            sA[ii][0] * sB[jj][0],
                            sA[ii][1] * sB[jj][1]
                        };
                        
                        __m256 tmp[2] = {
                            mul_sum_us8_pairs_float(b_pack_int8_low  [ jj ], a_pack_int8 [ ii ] [ 0 ] ),
                            mul_sum_us8_pairs_float(b_pack_int8_high [ jj ], a_pack_int8 [ ii ] [ 1 ] )
                        };
                        
                        sum_packed_fp[ii][jj] = _mm256_fmadd_ps(tmp[0], _mm256_set1_ps(fused_s[0]), sum_packed_fp[ii][jj]);
                        sum_packed_fp[ii][jj] = _mm256_fmadd_ps(tmp[1], _mm256_set1_ps(fused_s[1]), sum_packed_fp[ii][jj]);
                    
                        float sum0 = *ScaledSumA(i+ii, block_idx, K/Q_BLK_SIZE);
                        float sum1 = *ScaledSumA(i+ii, block_idx+1, K/Q_BLK_SIZE);
                        float min_b_0 = *MinB(block_idx, j+jj, K / Q_BLK_SIZE);
                        float min_b_1 = *MinB(block_idx + 1, j+jj, K / Q_BLK_SIZE);
                        extra_sum[ii][jj] += sum0 * min_b_0;
                        extra_sum[ii][jj] += sum1 * min_b_1;
                    }
                }
            }
            
            // Store results for all active i/j combinations
            int valid_m = min(M - i, 4);
            int valid_n = min(N - j, 4);
            for (int ii = 0; ii < valid_m; ++ii) {
                
                for (int jj = 0; jj < valid_n; ++jj) {
                    
                    *C(i+ii, j+jj, N) += hsum_float_8(sum_packed_fp[ii][jj]) + extra_sum[ii][jj];
                }
            }
        }
}



void matmul_kernel_gemv(
    int8_t* A, uint8_t* B, float* C,
    float* SA, float* ScaledSumA, float* SB, float* MinB,
    const int M, const int N, const int K) 
{

    PRAGMA_OMP_PARALLEL_FOR
    for (int j = 0; j < N - 1; j += 2)
    {
        __m256 sum_packed_fp_j0 = _mm256_setzero_ps();
        __m256 sum_packed_fp_j1 = _mm256_setzero_ps();
        float extra_sum_fp_j0 = 0;
        float extra_sum_fp_j1 = 0;
        for (int p = 0; p < K; p += 2 * Q_BLK_SIZE)
        {
            int block_idx = p / Q_BLK_SIZE;
            
            // Load scale factors for both j values
            float sA_0 = *SA(0, block_idx, K/Q_BLK_SIZE);
            float sB_0_j0 = *SB(block_idx, j, K/Q_BLK_SIZE);
            float sB_0_j1 = *SB(block_idx, j+1, K/Q_BLK_SIZE);
            float sA_1 = *SA(0, block_idx + 1, K/Q_BLK_SIZE);
            float sB_1_j0 = *SB(block_idx + 1, j, K/Q_BLK_SIZE);
            float sB_1_j1 = *SB(block_idx + 1, j+1, K/Q_BLK_SIZE);

            float fused_s_0_j0 = sA_0 * sB_0_j0;
            float fused_s_0_j1 = sA_0 * sB_0_j1;
            float fused_s_1_j0 = sA_1 * sB_1_j0;
            float fused_s_1_j1 = sA_1 * sB_1_j1;
            
            // Load min values and scaled sums for both j values
            float min_b_0_j0 = *MinB(block_idx, j, K/Q_BLK_SIZE);
            float min_b_0_j1 = *MinB(block_idx, j+1, K/Q_BLK_SIZE);
            float min_b_1_j0 = *MinB(block_idx + 1, j, K/Q_BLK_SIZE);
            float min_b_1_j1 = *MinB(block_idx + 1, j+1, K/Q_BLK_SIZE);
            float sum_a_0_fused = *ScaledSumA(0, block_idx, K/Q_BLK_SIZE);
            float sum_a_1_fused = *ScaledSumA(0, block_idx + 1, K/Q_BLK_SIZE);
            
            // Load and process input values
            __m256i a_pack_int8_low = _mm256_loadu_si256((__m256i *) A(0, p, K));
            __m256i a_pack_int8_high = _mm256_loadu_si256((__m256i *) A(0, p + Q_BLK_SIZE, K));
            __m256i b_pack_int8_j0 = _mm256_loadu_si256((__m256i *) B(p/2, j, K/2));
            __m256i b_pack_int8_j1 = _mm256_loadu_si256((__m256i *) B(p/2, j+1, K/2));

            __m256i mask = _mm256_set1_epi8(0x0f);
            __m256i b_pack_int8_low_j0 = _mm256_and_si256(b_pack_int8_j0, mask);
            __m256i b_pack_int8_high_j0 = _mm256_and_si256(_mm256_srli_epi16(b_pack_int8_j0, 4), mask);
            __m256i b_pack_int8_low_j1 = _mm256_and_si256(b_pack_int8_j1, mask);
            __m256i b_pack_int8_high_j1 = _mm256_and_si256(_mm256_srli_epi16(b_pack_int8_j1, 4), mask);

            // Compute products and accumulate for both j values
            __m256 tmp_0_j0 = mul_sum_us8_pairs_float(b_pack_int8_low_j0, a_pack_int8_low);
            __m256 tmp_1_j0 = mul_sum_us8_pairs_float(b_pack_int8_high_j0, a_pack_int8_high);
            __m256 tmp_0_j1 = mul_sum_us8_pairs_float(b_pack_int8_low_j1, a_pack_int8_low);
            __m256 tmp_1_j1 = mul_sum_us8_pairs_float(b_pack_int8_high_j1, a_pack_int8_high);

            sum_packed_fp_j0 = _mm256_fmadd_ps(tmp_0_j0, _mm256_set1_ps(fused_s_0_j0), sum_packed_fp_j0);
            sum_packed_fp_j0 = _mm256_fmadd_ps(tmp_1_j0, _mm256_set1_ps(fused_s_1_j0), sum_packed_fp_j0);
            sum_packed_fp_j1 = _mm256_fmadd_ps(tmp_0_j1, _mm256_set1_ps(fused_s_0_j1), sum_packed_fp_j1);
            sum_packed_fp_j1 = _mm256_fmadd_ps(tmp_1_j1, _mm256_set1_ps(fused_s_1_j1), sum_packed_fp_j1);

            extra_sum_fp_j0 += sum_a_0_fused * min_b_0_j0 + sum_a_1_fused * min_b_1_j0;
            extra_sum_fp_j1 += sum_a_0_fused * min_b_0_j1 + sum_a_1_fused * min_b_1_j1;
        }

        *C(0, j, N) += hsum_float_8(sum_packed_fp_j0) + extra_sum_fp_j0;
        *C(0, j+1, N) += hsum_float_8(sum_packed_fp_j1) + extra_sum_fp_j1;
    }

}
void MatmulOperator::qgemv_A8W8_kernel(struct qwen_matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32

    // quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);
    quantize_fp32_to_int8_q81(A->data_ptr, A->int8_data_ptr, params->A_scales, params->A_scaled_sum, A->row * A->column, block_size);


    int m = C->row, n = C->column, k = A->column;
    // zero out C
    for (int i = 0; i < m * n; i++) {
        C->data_ptr[i] = 0;
    }
    matmul_kernel_gemv(
        A->int8_data_ptr, B->int4_data_ptr, C->data_ptr,
        //Ink: We use offset to hold the minimum value of blocks in q4_1, for weights.
        params->A_scales, params->A_scaled_sum, params->scales, params->offset,
        m, n, k
    );


};
    
void MatmulOperator::qgemm_A8W4_kernel(struct qwen_matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32

    // quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);
    quantize_fp32_to_int8_q81(A->data_ptr, A->int8_data_ptr, params->A_scales, params->A_scaled_sum, A->row * A->column, block_size);


    int m = C->row, n = C->column, k = A->column;
    // zero out C
    for (int i = 0; i < m * n; i++) {
        C->data_ptr[i] = 0;
    }
    matmul_kernel(
        A->int8_data_ptr, B->int4_data_ptr, C->data_ptr,
        //Ink: We use offset to hold the minimum value of blocks in q4_1, for weights.
        params->A_scales, params->A_scaled_sum, params->scales, params->offset,
        m, n, k
    );


};
}  // namespace matmul
