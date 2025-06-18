#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

#include "QwenOperator.h"
#include "lib.h"
#include "operators.h"
#include <omp.h>

#define MEM_ALIGN 64
#define UN_INIT -10.0

#ifndef NTHREADS
#define NTHREADS 16
#endif

// openmp setting
#define OMP_SCHEDULE dynamic
#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")
#define NTHREADS_GEMV 4
#define PRAGMA_OMP_PARALLEL_FOR_GEMV _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS_GEMV)")

#define MR 6
#define NR 16

#define MC (6 * (800 / NTHREADS) * NTHREADS)
#define NC (16 * (40 / NTHREADS) * NTHREADS)
#define KC 500

#define MEM_ALIGN 64


#ifndef NITER
#define NITER 100
#endif

#define min(a, b) ((a) < (b) ? (a) : (b))

// row-major order: matrix A and C
// column-major order: matrix B
// A: M x K B: K x N C: M x N
#define A(i, j) (A + (i) * K + (j))
#define B(i, j) (B + (j) * K + (i))
#define C(i, j) (C + (i) * N + (j))

// column-major shape (block_m, K)
#define blockA(i, j) (blockA + (j) * block_m + (i))
// row-major shape (K, block_n)
#define blockB(i, j) (blockB + (i) * block_n + (j))
static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[KC * NC] __attribute__((aligned(64)));

namespace {

/*
blockA should be column-major order
blockB should be row-major order
*/
void kernel_16x6(float* blockA, float* blockB, float* C, const int valid_m, const int valid_n, const int block_m,
                 const int block_n, const int block_K, const int N) {
    // Explicitly unroll the array
    // __m256 C_buffer[6][2];
    __m256 C00 = _mm256_setzero_ps();
    __m256 C01 = _mm256_setzero_ps();
    __m256 C10 = _mm256_setzero_ps();
    __m256 C11 = _mm256_setzero_ps();
    __m256 C20 = _mm256_setzero_ps();
    __m256 C21 = _mm256_setzero_ps();
    __m256 C30 = _mm256_setzero_ps();
    __m256 C31 = _mm256_setzero_ps();
    __m256 C40 = _mm256_setzero_ps();
    __m256 C41 = _mm256_setzero_ps();
    __m256 C50 = _mm256_setzero_ps();
    __m256 C51 = _mm256_setzero_ps();

    __m256 a_packFloat8;
    __m256 b0_packFloat8;
    __m256 b1_packFloat8;
    // __m256i masks[2];
    // Explicitly unroll the array
    __m256i mask0;
    __m256i mask1;
    // load C from memory to registers
    // i: row index
    if (valid_n != NR) {
        static int8_t mask[32]
            __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
        mask0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - valid_n]));
        mask1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - valid_n + 8]));
        switch (valid_m) {
            case 6:
                C00 = _mm256_maskload_ps(C(0, 0), mask0);
                C01 = _mm256_maskload_ps(C(0, 8), mask1);
                C10 = _mm256_maskload_ps(C(1, 0), mask0);
                C11 = _mm256_maskload_ps(C(1, 8), mask1);
                C20 = _mm256_maskload_ps(C(2, 0), mask0);
                C21 = _mm256_maskload_ps(C(2, 8), mask1);
                C30 = _mm256_maskload_ps(C(3, 0), mask0);
                C31 = _mm256_maskload_ps(C(3, 8), mask1);
                C40 = _mm256_maskload_ps(C(4, 0), mask0);
                C41 = _mm256_maskload_ps(C(4, 8), mask1);
                C50 = _mm256_maskload_ps(C(5, 0), mask0);
                C51 = _mm256_maskload_ps(C(5, 8), mask1);
                break;
            case 5:
                C00 = _mm256_maskload_ps(C(0, 0), mask0);
                C01 = _mm256_maskload_ps(C(0, 8), mask1);
                C10 = _mm256_maskload_ps(C(1, 0), mask0);
                C11 = _mm256_maskload_ps(C(1, 8), mask1);
                C20 = _mm256_maskload_ps(C(2, 0), mask0);
                C21 = _mm256_maskload_ps(C(2, 8), mask1);
                C30 = _mm256_maskload_ps(C(3, 0), mask0);
                C31 = _mm256_maskload_ps(C(3, 8), mask1);
                C40 = _mm256_maskload_ps(C(4, 0), mask0);
                C41 = _mm256_maskload_ps(C(4, 8), mask1);
                break;
            case 4:
                C00 = _mm256_maskload_ps(C(0, 0), mask0);
                C01 = _mm256_maskload_ps(C(0, 8), mask1);
                C10 = _mm256_maskload_ps(C(1, 0), mask0);
                C11 = _mm256_maskload_ps(C(1, 8), mask1);
                C20 = _mm256_maskload_ps(C(2, 0), mask0);
                C21 = _mm256_maskload_ps(C(2, 8), mask1);
                C30 = _mm256_maskload_ps(C(3, 0), mask0);
                C31 = _mm256_maskload_ps(C(3, 8), mask1);
                break;
            case 3:
                C00 = _mm256_maskload_ps(C(0, 0), mask0);
                C01 = _mm256_maskload_ps(C(0, 8), mask1);
                C10 = _mm256_maskload_ps(C(1, 0), mask0);
                C11 = _mm256_maskload_ps(C(1, 8), mask1);
                C20 = _mm256_maskload_ps(C(2, 0), mask0);
                C21 = _mm256_maskload_ps(C(2, 8), mask1);
                break;
            case 2:
                C00 = _mm256_maskload_ps(C(0, 0), mask0);
                C01 = _mm256_maskload_ps(C(0, 8), mask1);
                C10 = _mm256_maskload_ps(C(1, 0), mask0);
                C11 = _mm256_maskload_ps(C(1, 8), mask1);
                break;
            case 1:
                C00 = _mm256_maskload_ps(C(0, 0), mask0);
                C01 = _mm256_maskload_ps(C(0, 8), mask1);
                break;
        }

    } else {
        switch (valid_m) {
            case 6:
                C00 = _mm256_loadu_ps(C(0, 0));
                C01 = _mm256_loadu_ps(C(0, 8));
                C10 = _mm256_loadu_ps(C(1, 0));
                C11 = _mm256_loadu_ps(C(1, 8));
                C20 = _mm256_loadu_ps(C(2, 0));
                C21 = _mm256_loadu_ps(C(2, 8));
                C30 = _mm256_loadu_ps(C(3, 0));
                C31 = _mm256_loadu_ps(C(3, 8));
                C40 = _mm256_loadu_ps(C(4, 0));
                C41 = _mm256_loadu_ps(C(4, 8));
                C50 = _mm256_loadu_ps(C(5, 0));
                C51 = _mm256_loadu_ps(C(5, 8));
                break;
            case 5:
                C00 = _mm256_loadu_ps(C(0, 0));
                C01 = _mm256_loadu_ps(C(0, 8));
                C10 = _mm256_loadu_ps(C(1, 0));
                C11 = _mm256_loadu_ps(C(1, 8));
                C20 = _mm256_loadu_ps(C(2, 0));
                C21 = _mm256_loadu_ps(C(2, 8));
                C30 = _mm256_loadu_ps(C(3, 0));
                C31 = _mm256_loadu_ps(C(3, 8));
                C40 = _mm256_loadu_ps(C(4, 0));
                C41 = _mm256_loadu_ps(C(4, 8));
                break;
            case 4:
                C00 = _mm256_loadu_ps(C(0, 0));
                C01 = _mm256_loadu_ps(C(0, 8));
                C10 = _mm256_loadu_ps(C(1, 0));
                C11 = _mm256_loadu_ps(C(1, 8));
                C20 = _mm256_loadu_ps(C(2, 0));
                C21 = _mm256_loadu_ps(C(2, 8));
                C30 = _mm256_loadu_ps(C(3, 0));
                C31 = _mm256_loadu_ps(C(3, 8));
                break;
            case 3:
                C00 = _mm256_loadu_ps(C(0, 0));
                C01 = _mm256_loadu_ps(C(0, 8));
                C10 = _mm256_loadu_ps(C(1, 0));
                C11 = _mm256_loadu_ps(C(1, 8));
                C20 = _mm256_loadu_ps(C(2, 0));
                C21 = _mm256_loadu_ps(C(2, 8));
                break;
            case 2:
                C00 = _mm256_loadu_ps(C(0, 0));
                C01 = _mm256_loadu_ps(C(0, 8));
                C10 = _mm256_loadu_ps(C(1, 0));
                C11 = _mm256_loadu_ps(C(1, 8));
                break;
            case 1:
                C00 = _mm256_loadu_ps(C(0, 0));
                C01 = _mm256_loadu_ps(C(0, 8));
                break;
        }
    }

    for (int p = 0; p < block_K; p++) {
        b0_packFloat8 = _mm256_loadu_ps(blockB(p, 0));
        b1_packFloat8 = _mm256_loadu_ps(blockB(p, 8));

        a_packFloat8 = _mm256_broadcast_ss(blockA(0, p));
        C00 = _mm256_fmadd_ps(a_packFloat8, b0_packFloat8, C00);
        C01 = _mm256_fmadd_ps(a_packFloat8, b1_packFloat8, C01);

        a_packFloat8 = _mm256_broadcast_ss(blockA(1, p));
        C10 = _mm256_fmadd_ps(a_packFloat8, b0_packFloat8, C10);
        C11 = _mm256_fmadd_ps(a_packFloat8, b1_packFloat8, C11);

        a_packFloat8 = _mm256_broadcast_ss(blockA(2, p));
        C20 = _mm256_fmadd_ps(a_packFloat8, b0_packFloat8, C20);
        C21 = _mm256_fmadd_ps(a_packFloat8, b1_packFloat8, C21);

        a_packFloat8 = _mm256_broadcast_ss(blockA(3, p));
        C30 = _mm256_fmadd_ps(a_packFloat8, b0_packFloat8, C30);
        C31 = _mm256_fmadd_ps(a_packFloat8, b1_packFloat8, C31);

        a_packFloat8 = _mm256_broadcast_ss(blockA(4, p));
        C40 = _mm256_fmadd_ps(a_packFloat8, b0_packFloat8, C40);
        C41 = _mm256_fmadd_ps(a_packFloat8, b1_packFloat8, C41);

        a_packFloat8 = _mm256_broadcast_ss(blockA(5, p));
        C50 = _mm256_fmadd_ps(a_packFloat8, b0_packFloat8, C50);
        C51 = _mm256_fmadd_ps(a_packFloat8, b1_packFloat8, C51);
    }
    if (valid_n != NR) {
        // for (int i = 0; i < valid_m; i++) {
        //     _mm256_maskstore_ps(C(i, 0), mask0, C_buffer[i][0]);
        //     _mm256_maskstore_ps(C(i, 8), mask1, C_buffer[i][1]);

        // }
        switch (valid_m) {
            case 6:
                _mm256_maskstore_ps(C(0, 0), mask0, C00);
                _mm256_maskstore_ps(C(0, 8), mask1, C01);
                _mm256_maskstore_ps(C(1, 0), mask0, C10);
                _mm256_maskstore_ps(C(1, 8), mask1, C11);
                _mm256_maskstore_ps(C(2, 0), mask0, C20);
                _mm256_maskstore_ps(C(2, 8), mask1, C21);
                _mm256_maskstore_ps(C(3, 0), mask0, C30);
                _mm256_maskstore_ps(C(3, 8), mask1, C31);
                _mm256_maskstore_ps(C(4, 0), mask0, C40);
                _mm256_maskstore_ps(C(4, 8), mask1, C41);
                _mm256_maskstore_ps(C(5, 0), mask0, C50);
                _mm256_maskstore_ps(C(5, 8), mask1, C51);
                break;
            case 5:
                _mm256_maskstore_ps(C(0, 0), mask0, C00);
                _mm256_maskstore_ps(C(0, 8), mask1, C01);
                _mm256_maskstore_ps(C(1, 0), mask0, C10);
                _mm256_maskstore_ps(C(1, 8), mask1, C11);
                _mm256_maskstore_ps(C(2, 0), mask0, C20);
                _mm256_maskstore_ps(C(2, 8), mask1, C21);
                _mm256_maskstore_ps(C(3, 0), mask0, C30);
                _mm256_maskstore_ps(C(3, 8), mask1, C31);
                _mm256_maskstore_ps(C(4, 0), mask0, C40);
                _mm256_maskstore_ps(C(4, 8), mask1, C41);
                break;
            case 4:
                _mm256_maskstore_ps(C(0, 0), mask0, C00);
                _mm256_maskstore_ps(C(0, 8), mask1, C01);
                _mm256_maskstore_ps(C(1, 0), mask0, C10);
                _mm256_maskstore_ps(C(1, 8), mask1, C11);
                _mm256_maskstore_ps(C(2, 0), mask0, C20);
                _mm256_maskstore_ps(C(2, 8), mask1, C21);
                _mm256_maskstore_ps(C(3, 0), mask0, C30);
                _mm256_maskstore_ps(C(3, 8), mask1, C31);
                break;
            case 3:
                _mm256_maskstore_ps(C(0, 0), mask0, C00);
                _mm256_maskstore_ps(C(0, 8), mask1, C01);
                _mm256_maskstore_ps(C(1, 0), mask0, C10);
                _mm256_maskstore_ps(C(1, 8), mask1, C11);
                _mm256_maskstore_ps(C(2, 0), mask0, C20);
                _mm256_maskstore_ps(C(2, 8), mask1, C21);
                break;
            case 2:
                _mm256_maskstore_ps(C(0, 0), mask0, C00);
                _mm256_maskstore_ps(C(0, 8), mask1, C01);
                _mm256_maskstore_ps(C(1, 0), mask0, C10);
                _mm256_maskstore_ps(C(1, 8), mask1, C11);
                break;
            case 1:
                _mm256_maskstore_ps(C(0, 0), mask0, C00);
                _mm256_maskstore_ps(C(0, 8), mask1, C01);
                break;
        }
    } else {
        // for (int i = 0; i < valid_m; i++) {
        //     _mm256_storeu_ps(C(i, 0), C_buffer[i][0]);
        //     _mm256_storeu_ps(C(i, 8), C_buffer[i][1]);
        // }
        switch (valid_m) {
            case 6:
                _mm256_storeu_ps(C(0, 0), C00);
                _mm256_storeu_ps(C(0, 8), C01);
                _mm256_storeu_ps(C(1, 0), C10);
                _mm256_storeu_ps(C(1, 8), C11);
                _mm256_storeu_ps(C(2, 0), C20);
                _mm256_storeu_ps(C(2, 8), C21);
                _mm256_storeu_ps(C(3, 0), C30);
                _mm256_storeu_ps(C(3, 8), C31);
                _mm256_storeu_ps(C(4, 0), C40);
                _mm256_storeu_ps(C(4, 8), C41);
                _mm256_storeu_ps(C(5, 0), C50);
                _mm256_storeu_ps(C(5, 8), C51);
                break;
            case 5:
                _mm256_storeu_ps(C(0, 0), C00);
                _mm256_storeu_ps(C(0, 8), C01);
                _mm256_storeu_ps(C(1, 0), C10);
                _mm256_storeu_ps(C(1, 8), C11);
                _mm256_storeu_ps(C(2, 0), C20);
                _mm256_storeu_ps(C(2, 8), C21);
                _mm256_storeu_ps(C(3, 0), C30);
                _mm256_storeu_ps(C(3, 8), C31);
                _mm256_storeu_ps(C(4, 0), C40);
                _mm256_storeu_ps(C(4, 8), C41);
                break;
            case 4:
                _mm256_storeu_ps(C(0, 0), C00);
                _mm256_storeu_ps(C(0, 8), C01);
                _mm256_storeu_ps(C(1, 0), C10);
                _mm256_storeu_ps(C(1, 8), C11);
                _mm256_storeu_ps(C(2, 0), C20);
                _mm256_storeu_ps(C(2, 8), C21);
                _mm256_storeu_ps(C(3, 0), C30);
                _mm256_storeu_ps(C(3, 8), C31);
                break;
            case 3:
                _mm256_storeu_ps(C(0, 0), C00);
                _mm256_storeu_ps(C(0, 8), C01);
                _mm256_storeu_ps(C(1, 0), C10);
                _mm256_storeu_ps(C(1, 8), C11);
                _mm256_storeu_ps(C(2, 0), C20);
                _mm256_storeu_ps(C(2, 8), C21);
                break;
            case 2:
                _mm256_storeu_ps(C(0, 0), C00);
                _mm256_storeu_ps(C(0, 8), C01);
                _mm256_storeu_ps(C(1, 0), C10);
                _mm256_storeu_ps(C(1, 8), C11);
                break;
            case 1:
                _mm256_storeu_ps(C(0, 0), C00);
                _mm256_storeu_ps(C(0, 8), C01);
                break;
        }
    }
}

/*
Pack A into the memory layout required by the kernel

A: The origianl matrix A
K: The hidden dimension of the original matrix A, we need it to retrieve elements, since A is row-major.
blockA: The packed matrix A. blockA is column-major, with static shape: MC * KC. NOTE some parts may not be valid
valid_m: The valid row of blockA
valid_k: The valid column of blockA
the valid elements are in matrix of size (valid_m * valid_k)
*/
void pack_blockA_panel(float* A, float* blockA, const int kc, const int mc, const int K) {
    // int block_m = MR;
    for (int j = 0; j < kc; ++j) {
        for (int i = 0; i < mc; ++i) {
            *(blockA) = *A(i, j);
            blockA++;
            // *(blockA(i, j )) = *A(i, j);
        }
    }
}
void pack_blockA(float* A, float* blockA, const int mc, const int kc, const int K) {
    PRAGMA_OMP_PARALLEL_FOR
    for (int i = 0; i < mc; i += MR) {
        int valid_m = min(mc - i, MR);
        // printf("\e[31m[ERROR]\e[m The valid_m = %d\n", valid_m);
        // if (valid_m != MC)
        // {
        //     printf("\e[31m[ERROR]\e[m The valid_m is not MC\n");
        // }
        pack_blockA_panel(A(i, 0), blockA + i * kc, kc, valid_m, K);
    }
}

/*
Pack B into the memory layout required by the kernel
B: The origianl matrix B
K: The hidden dimension of the original matrix A, we need it to retrieve elements, since B is column-major.
blockB: The packed matrix B. blockA is column-major, with static shape: KC * NC. NOTE some parts may not be valid
valid_n: The valid cloumns of blockB
valid_k: The valid rows of blockA
The valid elements of blockB are from 0 to (valid_n * valid_k)
*/

void pack_blockB_panel(float* B, float* blockB, const int nc, const int kc, const int K) {
    // int block_n = NR;
    for (int i = 0; i < kc; i++) {
        for (int j = 0; j < nc; j++) {
            *(blockB++) = *B(i, j);
            // *(blockB( i, j )) = *B(i, j);
        }
    }
}

void pack_blockB(float* B, float* blockB, const int nc, const int kc, const int K) {
    PRAGMA_OMP_PARALLEL_FOR
    for (int jc = 0; jc < nc; jc += NR) {
        int valid_NR = min(nc - jc, NR);
        pack_blockB_panel(B(0, jc), blockB + jc * kc, valid_NR, kc, K);
    }
}

}
/*
j: column index
i: row index
p: hidden index
c: Cache level
r: register level
*/
void gemm_fp32_rcr(float* A, float* B, float* C, const int M, const int N, const int K) {
    for (int ic = 0; ic < M; ic += MC) {
        int valid_MC = min(M - ic, MC);
        for (int pc = 0; pc < K; pc += KC) {
            int valid_KC = min(K - pc, KC);
            pack_blockA(A(ic, pc), blockA_packed, valid_MC, valid_KC, K);
            for (int jc = 0; jc < N; jc += NC) {
                int valid_NC = min(N - jc, NC);
                pack_blockB(B(pc, jc), blockB_packed, valid_NC, valid_KC, K);
                PRAGMA_OMP_PARALLEL_FOR
                for (int ir = 0; ir < valid_MC; ir += MR) {
                    for (int jr = 0; jr < valid_NC; jr += NR) {
                        int valid_MR = min(valid_MC - ir, MR);
                        int valid_NR = min(valid_NC - jr, NR);
                        float* blockA_panel = blockA_packed + ir * valid_KC;
                        float* blockB_panel = blockB_packed + jr * valid_KC;
                        kernel_16x6(blockA_panel, blockB_panel, C(ic + ir, jc + jr), valid_MR, valid_NR, valid_MR,
                                    valid_NR, valid_KC, N);
                    }
                }
            }
        }
    }
}

void gemv_fp32_rcr_mt_impl_1(float* A, float* B, float* C, const int M, const int N, const int K) {
    // PRAGMA_OMP_PARALLEL_FOR
    // PRAGMA_OMP_PARALLEL_FOR_GEMV
    #pragma omp parallel for num_threads(NTHREADS_GEMV)
    for( int j = 0; j < N; j++)
    {
        __m256 acc = _mm256_setzero_ps();
        for( int p = 0; p < K; p+=8)
        {
            __m256 lhs = _mm256_loadu_ps(A(0, p));
            __m256 rhs = _mm256_loadu_ps(B(p, j));
            acc = _mm256_fmadd_ps(lhs, rhs, acc);
        }
        *C(0, j) = hsum_float_8(acc);
    }
}

void kernel_per_thread(float* A, float* B, float* C, const int M, const int N, const int K, const int start_col, const int end_col) {
    // PRAGMA_OMP_PARALLEL_FOR
    for( int j = start_col; j < end_col; j++)
    {
        __m256 acc = _mm256_setzero_ps();
        for( int p = 0; p < K; p+=8)
        {
            __m256 lhs = _mm256_loadu_ps(A(0, p));
            __m256 rhs = _mm256_loadu_ps(B(p, j));
            acc = _mm256_fmadd_ps(lhs, rhs, acc);
        }
        *C(0, j) = hsum_float_8(acc);
    }
}

void gemv_fp32_rcr_mt_impl_2(float* A, float* B, float* C, const int M, const int N, const int K)
{
    int actual_threads = 0;
    #pragma omp parallel num_threads(4)
    {
        
        // Check actual number of threads (only print from thread 0 to avoid multiple prints)
        #pragma omp single
        {
            actual_threads = omp_get_num_threads();
            // printf("Requested threads: %d, Actual threads: %d\n", NTHREADS, actual_threads);
        }
        int thread_idx = omp_get_thread_num();
        int labor_per_thread = (N + actual_threads - 1) / actual_threads;
        int start_col = (labor_per_thread) * thread_idx;
        int end_col = (labor_per_thread) * (thread_idx + 1);
        end_col = end_col > N? N: end_col;
        // printf("start: %d, end:%d\n", start_col, end_col);
        kernel_per_thread(A, B, C, M, N, K, start_col, end_col);
    }
}