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

#define NTHREADS_GEMV 4

#define MR 6
#define NR 16
// #define MC MR *256
// #define NC NR * 64
// #define MC MR * 64
// #define NC NR * 256
// #define KC 2000


#define MC (6 * (800 / NTHREADS) * NTHREADS)
#define NC (16 * (40 / NTHREADS) * NTHREADS)
#define KC 500

#define MEM_ALIGN 64

#define min(a, b) ((a) < (b) ? (a) : (b))

// row-major order: matrix A and C
// column-major order: matrix B
// A: M x K B: K x N C: M x N
#define A(i, j, ld) ( A + ( i ) * ld + ( j ) )
#define B(i, j, ld) ( B + ( i ) * ld + ( j ) )
#define C(i, j, ld) ( C + ( i ) * ld + ( j ) )

// column-major shape (block_m, K)
#define blockA(i, j) ( blockA + ( j ) * block_m + ( i ) )
// row-major shape (K, block_n)
#define blockB(i, j) ( blockB + ( i ) * block_n + ( j ) )



static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[KC * NC] __attribute__((aligned(64)));

// openmp setting
#define OMP_SCHEDULE dynamic
#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")

namespace{

/*
blockA should be column-major order
blockB should be row-major order
*/
void kernel_16x6(float* blockA, float* blockB, float* C, const int valid_m, const int valid_n, const int block_m, const int block_n, const int block_K, const int N) {
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
    if (valid_n != NR)
    {
        // const unsigned int bit_mask = 65535;
        // mask0 = _mm256_setr_epi32(bit_mask << (valid_n + 15),
        //                              bit_mask << (valid_n + 14),
        //                              bit_mask << (valid_n + 13),
        //                              bit_mask << (valid_n + 12),
        //                              bit_mask << (valid_n + 11),
        //                              bit_mask << (valid_n + 10),
        //                              bit_mask << (valid_n + 9),
        //                              bit_mask << (valid_n + 8));
        // mask1 = _mm256_setr_epi32(bit_mask << (valid_n + 7),
        //                              bit_mask << (valid_n + 6),
        //                              bit_mask << (valid_n + 5),
        //                              bit_mask << (valid_n + 4),
        //                              bit_mask << (valid_n + 3),
        //                              bit_mask << (valid_n + 2),
        //                              bit_mask << (valid_n + 1),
        //                              bit_mask <<  valid_n);

static int8_t mask[32]
    __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
mask0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - valid_n]));
mask1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - valid_n + 8]));
        // Since we unroll the array, we need to manually load the data
        // for (int i =  0; i < valid_m; i++)
        // {
        //     C_buffer[i][0] = _mm256_maskload_ps(C(i, 0), mask0);
        //     C_buffer[i][1] = _mm256_maskload_ps(C(i, 8), mask1);
        // }
        switch (valid_m)
        {
            case 6:
                C00 = _mm256_maskload_ps(C(0, 0, N), mask0);
                C01 = _mm256_maskload_ps(C(0, 8, N), mask1);
                C10 = _mm256_maskload_ps(C(1, 0, N), mask0);
                C11 = _mm256_maskload_ps(C(1, 8, N), mask1);
                C20 = _mm256_maskload_ps(C(2, 0, N), mask0);
                C21 = _mm256_maskload_ps(C(2, 8, N), mask1);
                C30 = _mm256_maskload_ps(C(3, 0, N), mask0);
                C31 = _mm256_maskload_ps(C(3, 8, N), mask1);
                C40 = _mm256_maskload_ps(C(4, 0, N), mask0);
                C41 = _mm256_maskload_ps(C(4, 8, N), mask1);
                C50 = _mm256_maskload_ps(C(5, 0, N), mask0);
                C51 = _mm256_maskload_ps(C(5, 8, N), mask1);
                break;
            case 5:
                C00 = _mm256_maskload_ps(C(0, 0, N), mask0);
                C01 = _mm256_maskload_ps(C(0, 8, N), mask1);
                C10 = _mm256_maskload_ps(C(1, 0, N), mask0);
                C11 = _mm256_maskload_ps(C(1, 8, N), mask1);
                C20 = _mm256_maskload_ps(C(2, 0, N), mask0);
                C21 = _mm256_maskload_ps(C(2, 8, N), mask1);
                C30 = _mm256_maskload_ps(C(3, 0, N), mask0);
                C31 = _mm256_maskload_ps(C(3, 8, N), mask1);
                C40 = _mm256_maskload_ps(C(4, 0, N), mask0);
                C41 = _mm256_maskload_ps(C(4, 8, N), mask1);
                break;
            case 4:
                C00 = _mm256_maskload_ps(C(0, 0, N), mask0);
                C01 = _mm256_maskload_ps(C(0, 8, N), mask1);
                C10 = _mm256_maskload_ps(C(1, 0, N), mask0);
                C11 = _mm256_maskload_ps(C(1, 8, N), mask1);
                C20 = _mm256_maskload_ps(C(2, 0, N), mask0);
                C21 = _mm256_maskload_ps(C(2, 8, N), mask1);
                C30 = _mm256_maskload_ps(C(3, 0, N), mask0);
                C31 = _mm256_maskload_ps(C(3, 8, N), mask1);
                break;
            case 3:
                C00 = _mm256_maskload_ps(C(0, 0, N), mask0);
                C01 = _mm256_maskload_ps(C(0, 8, N), mask1);
                C10 = _mm256_maskload_ps(C(1, 0, N), mask0);
                C11 = _mm256_maskload_ps(C(1, 8, N), mask1);
                C20 = _mm256_maskload_ps(C(2, 0, N), mask0);
                C21 = _mm256_maskload_ps(C(2, 8, N), mask1);
                break;
            case 2:
                C00 = _mm256_maskload_ps(C(0, 0, N), mask0);
                C01 = _mm256_maskload_ps(C(0, 8, N), mask1);
                C10 = _mm256_maskload_ps(C(1, 0, N), mask0);
                C11 = _mm256_maskload_ps(C(1, 8, N), mask1);
                break;
            case 1:
                C00 = _mm256_maskload_ps(C(0, 0, N), mask0);
                C01 = _mm256_maskload_ps(C(0, 8, N), mask1);
                break;
        }
                                     
    } 
    else
    {
        // for (int i = 0; i < valid_m; i++) {
        //     C_buffer[i][0] = _mm256_loadu_ps(C(i, 0));
        //     C_buffer[i][1] = _mm256_loadu_ps(C(i, 8));
        // }
        switch (valid_m)
        {
            case 6:
                C00 = _mm256_loadu_ps(C(0, 0, N));
                C01 = _mm256_loadu_ps(C(0, 8, N));
                C10 = _mm256_loadu_ps(C(1, 0, N));
                C11 = _mm256_loadu_ps(C(1, 8, N));
                C20 = _mm256_loadu_ps(C(2, 0, N));
                C21 = _mm256_loadu_ps(C(2, 8, N));
                C30 = _mm256_loadu_ps(C(3, 0, N));
                C31 = _mm256_loadu_ps(C(3, 8, N));
                C40 = _mm256_loadu_ps(C(4, 0, N));
                C41 = _mm256_loadu_ps(C(4, 8, N));
                C50 = _mm256_loadu_ps(C(5, 0, N));
                C51 = _mm256_loadu_ps(C(5, 8, N));
                break;
            case 5:
                C00 = _mm256_loadu_ps(C(0, 0, N));
                C01 = _mm256_loadu_ps(C(0, 8, N));
                C10 = _mm256_loadu_ps(C(1, 0, N));
                C11 = _mm256_loadu_ps(C(1, 8, N));
                C20 = _mm256_loadu_ps(C(2, 0, N));
                C21 = _mm256_loadu_ps(C(2, 8, N));
                C30 = _mm256_loadu_ps(C(3, 0, N));
                C31 = _mm256_loadu_ps(C(3, 8, N));
                C40 = _mm256_loadu_ps(C(4, 0, N));
                C41 = _mm256_loadu_ps(C(4, 8, N));
                break;
            case 4:
                C00 = _mm256_loadu_ps(C(0, 0, N));
                C01 = _mm256_loadu_ps(C(0, 8, N));
                C10 = _mm256_loadu_ps(C(1, 0, N));
                C11 = _mm256_loadu_ps(C(1, 8, N));
                C20 = _mm256_loadu_ps(C(2, 0, N));
                C21 = _mm256_loadu_ps(C(2, 8, N));
                C30 = _mm256_loadu_ps(C(3, 0, N));
                C31 = _mm256_loadu_ps(C(3, 8, N));
                break;
            case 3:
                C00 = _mm256_loadu_ps(C(0, 0, N));
                C01 = _mm256_loadu_ps(C(0, 8, N));
                C10 = _mm256_loadu_ps(C(1, 0, N));
                C11 = _mm256_loadu_ps(C(1, 8, N));
                C20 = _mm256_loadu_ps(C(2, 0, N));
                C21 = _mm256_loadu_ps(C(2, 8, N));
                break;
            case 2:
                C00 = _mm256_loadu_ps(C(0, 0, N));
                C01 = _mm256_loadu_ps(C(0, 8, N));
                C10 = _mm256_loadu_ps(C(1, 0, N));
                C11 = _mm256_loadu_ps(C(1, 8, N));
                break;
            case 1:
                C00 = _mm256_loadu_ps(C(0, 0, N));
                C01 = _mm256_loadu_ps(C(0, 8, N));
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
    if (valid_n != NR)
    {
        // for (int i = 0; i < valid_m; i++) {
        //     _mm256_maskstore_ps(C(i, 0), mask0, C_buffer[i][0]);
        //     _mm256_maskstore_ps(C(i, 8), mask1, C_buffer[i][1]);

        // }
        switch(valid_m)
        {
            case 6:
                _mm256_maskstore_ps(C(0, 0, N), mask0, C00);
                _mm256_maskstore_ps(C(0, 8, N), mask1, C01);
                _mm256_maskstore_ps(C(1, 0, N), mask0, C10);
                _mm256_maskstore_ps(C(1, 8, N), mask1, C11);
                _mm256_maskstore_ps(C(2, 0, N), mask0, C20);
                _mm256_maskstore_ps(C(2, 8, N), mask1, C21);
                _mm256_maskstore_ps(C(3, 0, N), mask0, C30);
                _mm256_maskstore_ps(C(3, 8, N), mask1, C31);
                _mm256_maskstore_ps(C(4, 0, N), mask0, C40);
                _mm256_maskstore_ps(C(4, 8, N), mask1, C41);
                _mm256_maskstore_ps(C(5, 0, N), mask0, C50);
                _mm256_maskstore_ps(C(5, 8, N), mask1, C51);
                break;
            case 5:
                _mm256_maskstore_ps(C(0, 0, N), mask0, C00);
                _mm256_maskstore_ps(C(0, 8, N), mask1, C01);
                _mm256_maskstore_ps(C(1, 0, N), mask0, C10);
                _mm256_maskstore_ps(C(1, 8, N), mask1, C11);
                _mm256_maskstore_ps(C(2, 0, N), mask0, C20);
                _mm256_maskstore_ps(C(2, 8, N), mask1, C21);
                _mm256_maskstore_ps(C(3, 0, N), mask0, C30);
                _mm256_maskstore_ps(C(3, 8, N), mask1, C31);
                _mm256_maskstore_ps(C(4, 0, N), mask0, C40);
                _mm256_maskstore_ps(C(4, 8, N), mask1, C41);
                break;
            case 4:
                _mm256_maskstore_ps(C(0, 0, N), mask0, C00);
                _mm256_maskstore_ps(C(0, 8, N), mask1, C01);
                _mm256_maskstore_ps(C(1, 0, N), mask0, C10);
                _mm256_maskstore_ps(C(1, 8, N), mask1, C11);
                _mm256_maskstore_ps(C(2, 0, N), mask0, C20);
                _mm256_maskstore_ps(C(2, 8, N), mask1, C21);
                _mm256_maskstore_ps(C(3, 0, N), mask0, C30);
                _mm256_maskstore_ps(C(3, 8, N), mask1, C31);
                break;
            case 3:
                _mm256_maskstore_ps(C(0, 0, N), mask0, C00);
                _mm256_maskstore_ps(C(0, 8, N), mask1, C01);
                _mm256_maskstore_ps(C(1, 0, N), mask0, C10);
                _mm256_maskstore_ps(C(1, 8, N), mask1, C11);
                _mm256_maskstore_ps(C(2, 0, N), mask0, C20);
                _mm256_maskstore_ps(C(2, 8, N), mask1, C21);
                break;
            case 2:
                _mm256_maskstore_ps(C(0, 0, N), mask0, C00);
                _mm256_maskstore_ps(C(0, 8, N), mask1, C01);
                _mm256_maskstore_ps(C(1, 0, N), mask0, C10);
                _mm256_maskstore_ps(C(1, 8, N), mask1, C11);
                break;
            case 1:
                _mm256_maskstore_ps(C(0, 0, N), mask0, C00);
                _mm256_maskstore_ps(C(0, 8, N), mask1, C01);
                break;
        }
    }
    else
    {
        // for (int i = 0; i < valid_m; i++) {
        //     _mm256_storeu_ps(C(i, 0), C_buffer[i][0]);
        //     _mm256_storeu_ps(C(i, 8), C_buffer[i][1]);
        // }
        switch(valid_m)
        {
            case 6:
                _mm256_storeu_ps(C(0, 0, N), C00);
                _mm256_storeu_ps(C(0, 8, N), C01);
                _mm256_storeu_ps(C(1, 0, N), C10);
                _mm256_storeu_ps(C(1, 8, N), C11);
                _mm256_storeu_ps(C(2, 0, N), C20);
                _mm256_storeu_ps(C(2, 8, N), C21);
                _mm256_storeu_ps(C(3, 0, N), C30);
                _mm256_storeu_ps(C(3, 8, N), C31);
                _mm256_storeu_ps(C(4, 0, N), C40);
                _mm256_storeu_ps(C(4, 8, N), C41);
                _mm256_storeu_ps(C(5, 0, N), C50);
                _mm256_storeu_ps(C(5, 8, N), C51);
                break;
            case 5:
                _mm256_storeu_ps(C(0, 0, N), C00);
                _mm256_storeu_ps(C(0, 8, N), C01);
                _mm256_storeu_ps(C(1, 0, N), C10);
                _mm256_storeu_ps(C(1, 8, N), C11);
                _mm256_storeu_ps(C(2, 0, N), C20);
                _mm256_storeu_ps(C(2, 8, N), C21);
                _mm256_storeu_ps(C(3, 0, N), C30);
                _mm256_storeu_ps(C(3, 8, N), C31);
                _mm256_storeu_ps(C(4, 0, N), C40);
                _mm256_storeu_ps(C(4, 8, N), C41);
                break;
            case 4:
                _mm256_storeu_ps(C(0, 0, N), C00);
                _mm256_storeu_ps(C(0, 8, N), C01);
                _mm256_storeu_ps(C(1, 0, N), C10);
                _mm256_storeu_ps(C(1, 8, N), C11);
                _mm256_storeu_ps(C(2, 0, N), C20);
                _mm256_storeu_ps(C(2, 8, N), C21);
                _mm256_storeu_ps(C(3, 0, N), C30);
                _mm256_storeu_ps(C(3, 8, N), C31);
                break;
            case 3:
                _mm256_storeu_ps(C(0, 0, N), C00);
                _mm256_storeu_ps(C(0, 8, N), C01);
                _mm256_storeu_ps(C(1, 0, N), C10);
                _mm256_storeu_ps(C(1, 8, N), C11);
                _mm256_storeu_ps(C(2, 0, N), C20);
                _mm256_storeu_ps(C(2, 8, N), C21);
                break;
            case 2:
                _mm256_storeu_ps(C(0, 0, N), C00);
                _mm256_storeu_ps(C(0, 8, N), C01);
                _mm256_storeu_ps(C(1, 0, N), C10);
                _mm256_storeu_ps(C(1, 8, N), C11);
                break;
            case 1:
                _mm256_storeu_ps(C(0, 0, N), C00);
                _mm256_storeu_ps(C(0, 8, N), C01);
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
void pack_blockA_panel(float* A, float* blockA, const int kc, const int mc, const int K)
{
    for( int j = 0; j < kc; ++j)
    {
        for ( int i = 0; i < mc; ++i)
        {
            *(blockA) = *A(i, j, K);
            blockA++;
        }
    }
}
void pack_blockA(float* A, float* blockA, const int mc, const int kc, const int K)
{
    PRAGMA_OMP_PARALLEL_FOR
    for (int i = 0; i < mc; i += MR)
    {
        int valid_m = min(mc - i, MR);
        pack_blockA_panel(
            A(i, 0, K),
            blockA + i * kc,
            kc,
            valid_m,
            K
        );
    }
} 

void pack_blockB_panel(float *B, float *blockB, const int nc, const int kc, const int N)
{
    for (int i = 0; i < kc; i++)
    {
        for (int j = 0; j < nc; j++)
        {
            *(blockB++) = *B(i, j, N);
        }
    }
}

void pack_blockB(float* B, float* blockB, const int nc, const int kc, const int N)
{   
    PRAGMA_OMP_PARALLEL_FOR
    for (int jc = 0; jc < nc; jc += NR)
    {
        int valid_NR = min(nc - jc, NR);
        pack_blockB_panel(
            B(0, jc, N),
            blockB + jc * kc, 
            valid_NR,
            kc,
            N
        );
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
void gemm_fp32_rrr(float* A, float* B, float* C, const int M, const int N, const int K) {
    for (int ic = 0; ic < M; ic += MC)
    {
        int valid_MC = min(M - ic, MC); 
        for (int pc = 0; pc < K; pc += KC)
        {
            int valid_KC = min(K - pc, KC);
            pack_blockA(
                A(ic, pc, K),
                blockA_packed,
                valid_MC,
                valid_KC,
                K
            );
            for (int jc = 0; jc < N; jc += NC)
            {
                int valid_NC = min(N - jc, NC);
                pack_blockB(
                    B(pc, jc, N),
                    blockB_packed,
                    valid_NC,
                    valid_KC,
                    N
                );
                PRAGMA_OMP_PARALLEL_FOR
                for (int ir = 0; ir < valid_MC; ir += MR)
                {
                    for (int jr = 0; jr < valid_NC; jr += NR)
                    {
                        int valid_MR = min(valid_MC - ir, MR);
                        int valid_NR = min(valid_NC - jr, NR);
                        float *blockA_panel = blockA_packed + ir * valid_KC;
                        float *blockB_panel = blockB_packed + jr * valid_KC;
                        kernel_16x6(blockA_panel, blockB_panel, C(ic + ir, jc + jr, N), valid_MR, valid_NR, valid_MR, valid_NR, valid_KC, N);
                    }
                }
            }
        }
    }
}

void gemv_kernel_per_thread(float* A, float* B, float* C, const int M, const int N, const int K, int start_col, int end_col) {

    for( int p = 0; p < K; p++)
    {
        for (int j = start_col; j < end_col; j+=8)
        {
            __m256 acc = _mm256_loadu_ps(C(0, j, N));
            __m256 lhs = _mm256_set1_ps(*A(0, p, K));
            __m256 rhs = _mm256_loadu_ps(B(p, j, N));
            acc = _mm256_fmadd_ps(lhs, rhs, acc);
            _mm256_storeu_ps(C(0, j, N), acc);
        }

    }
}

void gemv_fp32_rrr(float* A, float* B, float* C, const int M, const int N, const int K) {
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
        int start_col = (N / actual_threads) * thread_idx;
        int end_col = (N / actual_threads) * (thread_idx + 1);
        gemv_kernel_per_thread(A, B, C, M, N, K, start_col, end_col);
    }
}

void gemv_fp32_rrr_naive(float* A, float* B, float* C, const int M, const int N, const int K) {
    for (int p = 0; p < K; p++)
    {
        for (int j = 0; j < N; j++) 
        {
            *C(0, j, N) += *A(0, p, K) * *B(p, j, N);
        }
    }

}