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
    float* SA, float* SB,
    const int M, const int N, const int K) 
{

    // Process 4 rows and 4 columns at a time
    PRAGMA_OMP_PARALLEL_FOR
    for (int i = 0; i < M; i+=4) 
    for (int j = 0; j < N; j+=4)
    {
        // Accumulators for all combinations of i and j
        __m256 sum_packed_fp[4][4];
        // float sum_a_fused[4][4];
        // float min_b[4][4];
        for (int ii = 0; ii < 4; ++ii) {
            for (int jj = 0; jj < 4; ++jj) {
                sum_packed_fp[ii][jj] = _mm256_setzero_ps();
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
                // b_pack_int8_low[jj] = _mm256_and_si256(b_pack_int8[jj], mask) - 8;
                // b_pack_int8_high[jj] = _mm256_and_si256(_mm256_srli_epi16(b_pack_int8[jj], 4), mask) - 8;

                // b_pack_int8_low[jj] = _mm256_and_si256(b_pack_int8[jj], mask);
                // b_pack_int8_high[jj] = _mm256_and_si256(_mm256_srli_epi16(b_pack_int8[jj], 4), mask);
                // b_pack_int8_low[jj] = _mm256_sub_epi8(b_pack_int8_low[jj], _mm256_set1_epi8(8));
                // b_pack_int8_high[jj] = _mm256_sub_epi8(b_pack_int8_high[jj], _mm256_set1_epi8(8));
                
                b_pack_int8_low[jj] = _mm256_sub_epi8(
                    _mm256_and_si256(b_pack_int8[jj], mask),
                    _mm256_set1_epi8(8)
                );
                b_pack_int8_high[jj] = _mm256_sub_epi8(
                    _mm256_and_si256(_mm256_srli_epi16(b_pack_int8[jj], 4), mask),
                    _mm256_set1_epi8(8)
                );

            }

            // Compute all active i/j combinations
            for (int ii = 0; ii < 4; ++ii) {
                for (int jj = 0; jj < 4; jj += 2) {
                    // Unroll jj by 2
                    float fused_s0[2] = {
                        sA[ii][0] * sB[jj][0],
                        sA[ii][1] * sB[jj][1]
                    };
                    float fused_s1[2] = {
                        sA[ii][0] * sB[jj+1][0],
                        sA[ii][1] * sB[jj+1][1]
                    };

                    const __m256i ax_low  = _mm256_sign_epi8(a_pack_int8[ii][0], a_pack_int8[ii][0]);
                    const __m256i ax_high = _mm256_sign_epi8(a_pack_int8[ii][1], a_pack_int8[ii][1]);

                    // jj
                    const __m256i sy_low0  = _mm256_sign_epi8(b_pack_int8_low[jj], a_pack_int8[ii][0]);
                    const __m256i sy_high0 = _mm256_sign_epi8(b_pack_int8_high[jj], a_pack_int8[ii][1]);
                    __m256 tmp0[2] = {
                        mul_sum_us8_pairs_float(ax_low, sy_low0),
                        mul_sum_us8_pairs_float(ax_high, sy_high0)
                    };
                    sum_packed_fp[ii][jj] = _mm256_fmadd_ps(tmp0[0], _mm256_set1_ps(fused_s0[0]), sum_packed_fp[ii][jj]);
                    sum_packed_fp[ii][jj] = _mm256_fmadd_ps(tmp0[1], _mm256_set1_ps(fused_s0[1]), sum_packed_fp[ii][jj]);

                    // jj+1
                    const __m256i sy_low1  = _mm256_sign_epi8(b_pack_int8_low[jj+1], a_pack_int8[ii][0]);
                    const __m256i sy_high1 = _mm256_sign_epi8(b_pack_int8_high[jj+1], a_pack_int8[ii][1]);
                    __m256 tmp1[2] = {
                        mul_sum_us8_pairs_float(ax_low, sy_low1),
                        mul_sum_us8_pairs_float(ax_high, sy_high1)
                    };
                    sum_packed_fp[ii][jj+1] = _mm256_fmadd_ps(tmp1[0], _mm256_set1_ps(fused_s1[0]), sum_packed_fp[ii][jj+1]);
                    sum_packed_fp[ii][jj+1] = _mm256_fmadd_ps(tmp1[1], _mm256_set1_ps(fused_s1[1]), sum_packed_fp[ii][jj+1]);
                }
            }
        }

        // Store results for all active i/j combinations
        int valid_m = min(M-i, 4);
        int valid_n = min(N-j, 4);
        for (int ii = 0; ii < valid_m; ++ii) {

            for (int jj = 0; jj < valid_n; ++jj) {

                *C(i+ii, j+jj, N) += hsum_float_8(sum_packed_fp[ii][jj]);
            }
        }

    }
}



void matmul_kernel_gemv(
    int8_t* A, uint8_t* B, float* C,
    float* SA, float* SB, 
    const int M, const int N, const int K) 
{
    PRAGMA_OMP_PARALLEL_FOR
    for (int j = 0; j < N; j += 4)
    {
        __m256 sum_packed_fp_0 = _mm256_setzero_ps();
        __m256 sum_packed_fp_1 = _mm256_setzero_ps();
        __m256 sum_packed_fp_2 = _mm256_setzero_ps();
        __m256 sum_packed_fp_3 = _mm256_setzero_ps();
        
        for (int p = 0; p < K; p += 2 * Q_BLK_SIZE)
        {
            int block_idx = p / Q_BLK_SIZE;
            
            // Load scale factors for j, j+1, j+2, j+3
            float sA_0 = *SA(0, block_idx, K/Q_BLK_SIZE);
            float sB_0_j0 = *SB(block_idx, j,   K/Q_BLK_SIZE);
            float sB_0_j1 = *SB(block_idx, j+1, K/Q_BLK_SIZE);
            float sB_0_j2 = *SB(block_idx, j+2, K/Q_BLK_SIZE);
            float sB_0_j3 = *SB(block_idx, j+3, K/Q_BLK_SIZE);
            
            float sA_1 = *SA(0, block_idx + 1, K/Q_BLK_SIZE);
            float sB_1_j0 = *SB(block_idx + 1, j,   K/Q_BLK_SIZE);
            float sB_1_j1 = *SB(block_idx + 1, j+1, K/Q_BLK_SIZE);
            float sB_1_j2 = *SB(block_idx + 1, j+2, K/Q_BLK_SIZE);
            float sB_1_j3 = *SB(block_idx + 1, j+3, K/Q_BLK_SIZE);
            
            float fused_s_0_j0 = sA_0 * sB_0_j0;
            float fused_s_1_j0 = sA_1 * sB_1_j0;
            float fused_s_0_j1 = sA_0 * sB_0_j1;
            float fused_s_1_j1 = sA_1 * sB_1_j1;
            float fused_s_0_j2 = sA_0 * sB_0_j2;
            float fused_s_1_j2 = sA_1 * sB_1_j2;
            float fused_s_0_j3 = sA_0 * sB_0_j3;
            float fused_s_1_j3 = sA_1 * sB_1_j3;
            
            // Load A packed int8 (same for all j iterations)
            __m256i a_pack_int8_low = _mm256_loadu_si256((__m256i *) A(0, p, K));
            __m256i a_pack_int8_high = _mm256_loadu_si256((__m256i *) A(0, p + Q_BLK_SIZE, K));
            
            // Load B packed int8 for j, j+1, j+2, j+3
            __m256i b_pack_int8_j0 = _mm256_loadu_si256((__m256i *) B(p/2, j,   K/2));
            __m256i b_pack_int8_j1 = _mm256_loadu_si256((__m256i *) B(p/2, j+1, K/2));
            __m256i b_pack_int8_j2 = _mm256_loadu_si256((__m256i *) B(p/2, j+2, K/2));
            __m256i b_pack_int8_j3 = _mm256_loadu_si256((__m256i *) B(p/2, j+3, K/2));
            
            __m256i mask = _mm256_set1_epi8(0x0f);
            
            // Process B for j
            __m256i b_pack_int8_low_j0 = _mm256_and_si256(b_pack_int8_j0, mask);
            __m256i b_pack_int8_high_j0 = _mm256_and_si256(_mm256_srli_epi16(b_pack_int8_j0, 4), mask);
            b_pack_int8_low_j0 = _mm256_sub_epi8(b_pack_int8_low_j0, _mm256_set1_epi8(8));
            b_pack_int8_high_j0 = _mm256_sub_epi8(b_pack_int8_high_j0, _mm256_set1_epi8(8));
            
            // Process B for j+1
            __m256i b_pack_int8_low_j1 = _mm256_and_si256(b_pack_int8_j1, mask);
            __m256i b_pack_int8_high_j1 = _mm256_and_si256(_mm256_srli_epi16(b_pack_int8_j1, 4), mask);
            b_pack_int8_low_j1 = _mm256_sub_epi8(b_pack_int8_low_j1, _mm256_set1_epi8(8));
            b_pack_int8_high_j1 = _mm256_sub_epi8(b_pack_int8_high_j1, _mm256_set1_epi8(8));
            
            // Process B for j+2
            __m256i b_pack_int8_low_j2 = _mm256_and_si256(b_pack_int8_j2, mask);
            __m256i b_pack_int8_high_j2 = _mm256_and_si256(_mm256_srli_epi16(b_pack_int8_j2, 4), mask);
            b_pack_int8_low_j2 = _mm256_sub_epi8(b_pack_int8_low_j2, _mm256_set1_epi8(8));
            b_pack_int8_high_j2 = _mm256_sub_epi8(b_pack_int8_high_j2, _mm256_set1_epi8(8));
            
            // Process B for j+3
            __m256i b_pack_int8_low_j3 = _mm256_and_si256(b_pack_int8_j3, mask);
            __m256i b_pack_int8_high_j3 = _mm256_and_si256(_mm256_srli_epi16(b_pack_int8_j3, 4), mask);
            b_pack_int8_low_j3 = _mm256_sub_epi8(b_pack_int8_low_j3, _mm256_set1_epi8(8));
            b_pack_int8_high_j3 = _mm256_sub_epi8(b_pack_int8_high_j3, _mm256_set1_epi8(8));
            
            // Compute for j
            __m256 tmp_0_j0 = mul_sum_i8_pairs_float(a_pack_int8_low, b_pack_int8_low_j0);
            __m256 tmp_1_j0 = mul_sum_i8_pairs_float(a_pack_int8_high, b_pack_int8_high_j0);
            sum_packed_fp_0 = _mm256_fmadd_ps(tmp_0_j0, _mm256_set1_ps(fused_s_0_j0), sum_packed_fp_0);
            sum_packed_fp_0 = _mm256_fmadd_ps(tmp_1_j0, _mm256_set1_ps(fused_s_1_j0), sum_packed_fp_0);
            
            // Compute for j+1
            __m256 tmp_0_j1 = mul_sum_i8_pairs_float(a_pack_int8_low, b_pack_int8_low_j1);
            __m256 tmp_1_j1 = mul_sum_i8_pairs_float(a_pack_int8_high, b_pack_int8_high_j1);
            sum_packed_fp_1 = _mm256_fmadd_ps(tmp_0_j1, _mm256_set1_ps(fused_s_0_j1), sum_packed_fp_1);
            sum_packed_fp_1 = _mm256_fmadd_ps(tmp_1_j1, _mm256_set1_ps(fused_s_1_j1), sum_packed_fp_1);
            
            // Compute for j+2
            __m256 tmp_0_j2 = mul_sum_i8_pairs_float(a_pack_int8_low, b_pack_int8_low_j2);
            __m256 tmp_1_j2 = mul_sum_i8_pairs_float(a_pack_int8_high, b_pack_int8_high_j2);
            sum_packed_fp_2 = _mm256_fmadd_ps(tmp_0_j2, _mm256_set1_ps(fused_s_0_j2), sum_packed_fp_2);
            sum_packed_fp_2 = _mm256_fmadd_ps(tmp_1_j2, _mm256_set1_ps(fused_s_1_j2), sum_packed_fp_2);
            
            // Compute for j+3
            __m256 tmp_0_j3 = mul_sum_i8_pairs_float(a_pack_int8_low, b_pack_int8_low_j3);
            __m256 tmp_1_j3 = mul_sum_i8_pairs_float(a_pack_int8_high, b_pack_int8_high_j3);
            sum_packed_fp_3 = _mm256_fmadd_ps(tmp_0_j3, _mm256_set1_ps(fused_s_0_j3), sum_packed_fp_3);
            sum_packed_fp_3 = _mm256_fmadd_ps(tmp_1_j3, _mm256_set1_ps(fused_s_1_j3), sum_packed_fp_3);
        }
        
        // Store results with bounds checking
        *C(0, j, N) += hsum_float_8(sum_packed_fp_0);
        *C(0, j+1, N) += hsum_float_8(sum_packed_fp_1);
        *C(0, j+2, N) += hsum_float_8(sum_packed_fp_2);
        *C(0, j+3, N) += hsum_float_8(sum_packed_fp_3);
    }
}

void MatmulOperator::qgemv_A8W8_kernel(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32

    quantize_fp32_to_int8_q80(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);
    // quantize_fp32_to_int8_q8_1(A->data_ptr, A->int8_data_ptr, params->A_scales, params->A_scaled_sum, A->row * A->column, block_size);


    int m = C->row, n = C->column, k = A->column;
    // zero out C
    for (int i = 0; i < m * n; i++) {
        C->data_ptr[i] = 0;
    }
    matmul_kernel_gemv(
        A->int8_data_ptr, B->int4_data_ptr, C->data_ptr,
        //Ink: We use offset to hold the minimum value of blocks in q4_1, for weights.
        params->A_scales, params->scales, 
        m, n, k
    );


};
    
void MatmulOperator::qgemm_A8W4_kernel(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32

    quantize_fp32_to_int8_q80(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);
    // quantize_fp32_to_int8_q8_1(A->data_ptr, A->int8_data_ptr, params->A_scales, params->A_scaled_sum, A->row * A->column, block_size);


    int m = C->row, n = C->column, k = A->column;
    // zero out C
    for (int i = 0; i < m * n; i++) {
        C->data_ptr[i] = 0;
    }
    matmul_kernel(
        A->int8_data_ptr, B->int4_data_ptr, C->data_ptr,
        //Ink: We use offset to hold the minimum value of blocks in q4_1, for weights.
        params->A_scales, params->scales,
        m, n, k
    );


};
}  // namespace matmul
