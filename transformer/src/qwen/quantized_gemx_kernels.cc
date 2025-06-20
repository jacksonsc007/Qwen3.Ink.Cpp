#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include "QwenOperator.h"
#include "ggml-impl.h"
#include "lib.h"
#include "operators.h"

#define Q_BLK_SIZE 32
#define QK8_0 32
#define QK8_1 32

#ifndef NTHREADS
#define NTHREADS 16
#endif

#define OMP_SCHEDULE dynamic
#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")
#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")


#define min(a, b) ((a) < (b) ? (a) : (b))

// row-major order: matrix A and C, SA
// column-major order: matrix B, SB
// A: M x K B: K x N C: M x N

#define A(i, j, ld) ( A + ( i ) * ( ld ) + ( j ) )
#define B(i, j, ld) ( B + ( j ) * ( ld ) + ( i ) )
#define C(i, j, ld) ( C + ( i ) * ( ld ) + ( j ) )
#define SA(i, j, ld) (SA + (i) * (ld) + (j))
#define SB(i, j, ld) (SB + (j) * (ld) + (i))

#define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define BROADCAST_FP16_FP32(x)  _mm256_cvtph_ps(_mm_set1_epi16(x))

void gemv_repack_A80W40(
    void * A_repack,
    void * B_repack,
    float* C,
    const int M, const int N, const int K
)
{

    struct q4_repack_2x8_fp16 * B_ptr_start = (struct q4_repack_2x8_fp16 *) B_repack;
    struct q8_repack_1x2_fp16 * A_ptr_start = (struct q8_repack_1x2_fp16 *) A_repack;
    // how many q4_repack and q8_repack blocks along the N dimension
    int nb_k = K / (2 * Q_BLK_SIZE);
    // how many q4_repack blocks along the K dimension
    int nb_n = N / 8; 
    // how many q8_repack blocks along the M dimension
    int nb_m = M ;
    {
        PRAGMA_OMP_PARALLEL_FOR
        for(int j = 0; j < nb_n; j++)
        {
            struct q4_repack_2x8_fp16 * B_ptr = B_ptr_start + j * nb_k;
            struct q8_repack_1x2_fp16 * A_ptr = A_ptr_start + 0 * nb_k;
            __m256 acc_row = _mm256_setzero_ps();
            for (int blk_id = 0; blk_id < nb_k; blk_id++)
            {
                struct q4_repack_2x8_fp16 B_block =  B_ptr[blk_id];
                // each pack contains 4 elements in the quantization block
                __m256i low_blk_B_packs[8];
                __m256i high_blk_B_packs[8];
                for (int pack_idx = 0; pack_idx < 8; pack_idx+=1)
                {
                    __m256i B_coupled_pack = _mm256_loadu_si256((__m256i *) (B_block.q_coupled + pack_idx * 32));
                    __m256i low_blk_B_pack = _mm256_and_si256(B_coupled_pack, _mm256_set1_epi8(0x0f));
                    low_blk_B_pack = _mm256_sub_epi8(low_blk_B_pack, _mm256_set1_epi8(8));
                    __m256i high_blk_B_pack = _mm256_and_si256(_mm256_srli_epi16(B_coupled_pack, 4), _mm256_set1_epi8(0x0f));
                    high_blk_B_pack = _mm256_sub_epi8(high_blk_B_pack, _mm256_set1_epi8(8));
                    low_blk_B_packs[pack_idx] = low_blk_B_pack;
                    high_blk_B_packs[pack_idx] = high_blk_B_pack;
                }
                __m256i low_blk_A_packs[8];
                __m256i high_blk_A_packs[8];
                for (int pack_idx = 0; pack_idx < 8; pack_idx+=1)
                {
                    low_blk_A_packs[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr[blk_id].q_low) )[pack_idx]);
                    high_blk_A_packs[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr[blk_id].q_high) )[pack_idx]);
                }
                __m256i iacc_low = _mm256_setzero_si256();
                __m256i iacc_high = _mm256_setzero_si256();
                for (int pack_idx = 0; pack_idx < 8; pack_idx+=1)
                {
                    iacc_low = mul_sum_i8_pairs_acc_int32x8(iacc_low, low_blk_A_packs[pack_idx], low_blk_B_packs[pack_idx]);
                    iacc_high = mul_sum_i8_pairs_acc_int32x8(iacc_high, high_blk_A_packs[pack_idx], high_blk_B_packs[pack_idx]);
                }
                // __m256i iacc = _mm256_add_epi32(iacc_low, iacc_high);
                
                // load scaleing factors
                __m256 sB_low = GGML_F32Cx8_LOAD(B_block.s_low);
                __m256 sB_high = GGML_F32Cx8_LOAD(B_block.s_high);
                __m256 sA_low = BROADCAST_FP16_FP32(A_ptr[blk_id].s_low);
                __m256 sA_high = BROADCAST_FP16_FP32(A_ptr[blk_id].s_high);
                __m256 s_low = _mm256_mul_ps(sA_low, sB_low);
                __m256 s_high = _mm256_mul_ps(sA_high, sB_high);
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low), s_low, acc_row);
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high), s_high, acc_row);

            }
            _mm256_storeu_ps(C(0, j * 8, N), acc_row);
        }
    }
}


void gemv_repack_A81W41_fp16(
    void * A_repack,
    void * B_repack,
    float* C,
    const int M, const int N, const int K
)
{
    struct q4_repack_2x8_fp16 * B_ptr_start = (struct q4_repack_2x8_fp16 *) B_repack;
    struct q8_repack_1x2_fp16 * A_ptr_start = (struct q8_repack_1x2_fp16 *) A_repack;
    // how many q4_repack and q8_repack blocks along the K dimension
    int nb_k = K / (2 * Q_BLK_SIZE);
    // how many q4_repack blocks along the N dimension
    int nb_n = N / 8; 
    
    // Process single row of A
    struct q8_repack_1x2_fp16 * A_ptr = A_ptr_start;

    PRAGMA_OMP_PARALLEL_FOR
    for(int j = 0; j < nb_n; j++)
    {
        struct q4_repack_2x8_fp16 * B_ptr = B_ptr_start + j * nb_k;
        
        __m256 acc_row = _mm256_setzero_ps();
        
        for (int blk_id = 0; blk_id < nb_k; blk_id++)
        {
            struct q4_repack_2x8_fp16 B_block = B_ptr[blk_id];
            // each pack contains 4 elements in the quantization block
            __m256i low_blk_B_packs[8];
            __m256i high_blk_B_packs[8];
            for (int pack_idx = 0; pack_idx < 8; pack_idx+=1)
            {
                __m256i B_coupled_pack = _mm256_loadu_si256((__m256i *) (B_block.q_coupled + pack_idx * 32));
                __m256i low_blk_B_pack = _mm256_and_si256(B_coupled_pack, _mm256_set1_epi8(0x0f));
                __m256i high_blk_B_pack = _mm256_and_si256(_mm256_srli_epi16(B_coupled_pack, 4), _mm256_set1_epi8(0x0f));
                low_blk_B_packs[pack_idx] = low_blk_B_pack;
                high_blk_B_packs[pack_idx] = high_blk_B_pack;
            }
            
            // Process single row of A
            __m256i low_blk_A_packs[8];
            __m256i high_blk_A_packs[8];
            
            for (int pack_idx = 0; pack_idx < 8; pack_idx+=1)
            {
                low_blk_A_packs[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr[blk_id].q_low) )[pack_idx]);
                high_blk_A_packs[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr[blk_id].q_high) )[pack_idx]);
            }
            
            __m256i iacc_low = _mm256_setzero_si256();
            __m256i iacc_high = _mm256_setzero_si256();
            
            for (int pack_idx = 0; pack_idx < 8; pack_idx+=1)
            {
                iacc_low  = mul_sum_us8_pairs_acc_int32x8 ( iacc_low , low_blk_B_packs  [ pack_idx ] , low_blk_A_packs  [ pack_idx ]); 
                iacc_high = mul_sum_us8_pairs_acc_int32x8 ( iacc_high, high_blk_B_packs [ pack_idx ] , high_blk_A_packs [ pack_idx ]); 
            }
            
            // load scaling factors
            __m256 sB_low    = GGML_F32Cx8_LOAD ( B_block.s_low    ); 
            __m256 sB_high   = GGML_F32Cx8_LOAD ( B_block.s_high   ); 
            __m256 minB_low  = GGML_F32Cx8_LOAD ( B_block.min_low  ); 
            __m256 minB_high = GGML_F32Cx8_LOAD ( B_block.min_high ); 

            __m256 sA_low          = BROADCAST_FP16_FP32 ( A_ptr [ blk_id ] .s_low           ); 
            __m256 sA_high         = BROADCAST_FP16_FP32 ( A_ptr [ blk_id ] .s_high          ); 
            __m256 scaled_sum_low  = BROADCAST_FP16_FP32 ( A_ptr [ blk_id ] .scaled_sum_low  ); 
            __m256 scaled_sum_high = BROADCAST_FP16_FP32 ( A_ptr [ blk_id ] .scaled_sum_high ); 

            __m256 s_low = _mm256_mul_ps(sA_low, sB_low);
            __m256 s_high = _mm256_mul_ps(sA_high, sB_high);

            // Accumulate results
            acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low), s_low, acc_row);
            acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high), s_high, acc_row);
            acc_row = _mm256_fmadd_ps(minB_low, scaled_sum_low, acc_row);
            acc_row = _mm256_fmadd_ps(minB_high, scaled_sum_high, acc_row);
        }
        
        _mm256_storeu_ps(C + j * 8, acc_row);
    }
}

void gemm_repack_A80W40(
    void * A_repack,
    void * B_repack,
    float* C,
    const int M, const int N, const int K
)
{
    // printf("\e[31m[num_threads = %d]\e[m \n", NTHREADS);
    struct q4_repack_2x8_fp16 * B_ptr_start = (struct q4_repack_2x8_fp16 *) B_repack;
    struct q8_repack_1x2_fp16 * A_ptr_start = (struct q8_repack_1x2_fp16 *) A_repack;
    // how many q4_repack and q8_repack blocks along the N dimension
    int nb_k = K / (2 * Q_BLK_SIZE);
    // how many q4_repack blocks along the K dimension
    int nb_n = N / 8; 
    // how many q8_repack blocks along the M dimension
    int nb_m = M;
    PRAGMA_OMP_PARALLEL_FOR
    for (int i = 0; i < nb_m; i += 2)
    {
        int valid_i = min(nb_m - i, 2);
        for(int j = 0; j < nb_n; j += 2)
        {
            // Handle 2x2 block of output
            struct q4_repack_2x8_fp16 * B_ptr_0 = B_ptr_start + j * nb_k;
            struct q4_repack_2x8_fp16 * B_ptr_1 = B_ptr_start + (j + 1) * nb_k;
            struct q8_repack_1x2_fp16 * A_ptr_0 = A_ptr_start + i * nb_k;
            struct q8_repack_1x2_fp16 * A_ptr_1 = A_ptr_start + (i + 1) * nb_k;

            // Initialize 4 accumulators for 2x2 block
            __m256 acc_row_00 = _mm256_setzero_ps();
            __m256 acc_row_01 = _mm256_setzero_ps();
            __m256 acc_row_10 = _mm256_setzero_ps();
            __m256 acc_row_11 = _mm256_setzero_ps();

            for (int blk_id = 0; blk_id < nb_k; blk_id++)
            {
                // Load B blocks
                struct q4_repack_2x8_fp16 B_block_0 = B_ptr_0[blk_id];
                struct q4_repack_2x8_fp16 B_block_1 = B_ptr_1[blk_id];

                // Process B blocks - low and high parts
                __m256i low_blk_B_packs_0[8];
                __m256i high_blk_B_packs_0[8];
                __m256i low_blk_B_packs_1[8];
                __m256i high_blk_B_packs_1[8];

                for (int pack_idx = 0; pack_idx < 8; pack_idx++)
                {
                    // Process first B block
                    __m256i B_coupled_pack_0 = _mm256_loadu_si256((__m256i *) (B_block_0.q_coupled + pack_idx * 32));
                    __m256i low_blk_B_pack_0 = _mm256_and_si256(B_coupled_pack_0, _mm256_set1_epi8(0x0f));
                    low_blk_B_pack_0 = _mm256_sub_epi8(low_blk_B_pack_0, _mm256_set1_epi8(8));
                    __m256i high_blk_B_pack_0 = _mm256_and_si256(_mm256_srli_epi16(B_coupled_pack_0, 4), _mm256_set1_epi8(0x0f));
                    high_blk_B_pack_0 = _mm256_sub_epi8(high_blk_B_pack_0, _mm256_set1_epi8(8));
                    low_blk_B_packs_0[pack_idx] = low_blk_B_pack_0;
                    high_blk_B_packs_0[pack_idx] = high_blk_B_pack_0;

                    // Process second B block
                    __m256i B_coupled_pack_1 = _mm256_loadu_si256((__m256i *) (B_block_1.q_coupled + pack_idx * 32));
                    __m256i low_blk_B_pack_1 = _mm256_and_si256(B_coupled_pack_1, _mm256_set1_epi8(0x0f));
                    low_blk_B_pack_1 = _mm256_sub_epi8(low_blk_B_pack_1, _mm256_set1_epi8(8));
                    __m256i high_blk_B_pack_1 = _mm256_and_si256(_mm256_srli_epi16(B_coupled_pack_1, 4), _mm256_set1_epi8(0x0f));
                    high_blk_B_pack_1 = _mm256_sub_epi8(high_blk_B_pack_1, _mm256_set1_epi8(8));
                    low_blk_B_packs_1[pack_idx] = low_blk_B_pack_1;
                    high_blk_B_packs_1[pack_idx] = high_blk_B_pack_1;
                }

                // Load A blocks
                __m256i low_blk_A_packs_0[8];
                __m256i high_blk_A_packs_0[8];
                __m256i low_blk_A_packs_1[8];
                __m256i high_blk_A_packs_1[8];

                for (int pack_idx = 0; pack_idx < 8; pack_idx++)
                {
                    low_blk_A_packs_0[pack_idx] = _mm256_set1_epi32(((int32_t *)(A_ptr_0[blk_id].q_low))[pack_idx]);
                    high_blk_A_packs_0[pack_idx] = _mm256_set1_epi32(((int32_t *)(A_ptr_0[blk_id].q_high))[pack_idx]);
                    low_blk_A_packs_1[pack_idx] = _mm256_set1_epi32(((int32_t *)(A_ptr_1[blk_id].q_low))[pack_idx]);
                    high_blk_A_packs_1[pack_idx] = _mm256_set1_epi32(((int32_t *)(A_ptr_1[blk_id].q_high))[pack_idx]);
                }

                // Initialize accumulators for this block
                __m256i iacc_low_00 = _mm256_setzero_si256();
                __m256i iacc_high_00 = _mm256_setzero_si256();
                __m256i iacc_low_01 = _mm256_setzero_si256();
                __m256i iacc_high_01 = _mm256_setzero_si256();
                __m256i iacc_low_10 = _mm256_setzero_si256();
                __m256i iacc_high_10 = _mm256_setzero_si256();
                __m256i iacc_low_11 = _mm256_setzero_si256();
                __m256i iacc_high_11 = _mm256_setzero_si256();

                // Compute dot products
                for (int pack_idx = 0; pack_idx < 8; pack_idx++)
                {
                    // First row of output
                    iacc_low_00 = mul_sum_i8_pairs_acc_int32x8(iacc_low_00, low_blk_A_packs_0[pack_idx], low_blk_B_packs_0[pack_idx]);
                    iacc_high_00 = mul_sum_i8_pairs_acc_int32x8(iacc_high_00, high_blk_A_packs_0[pack_idx], high_blk_B_packs_0[pack_idx]);
                    iacc_low_01 = mul_sum_i8_pairs_acc_int32x8(iacc_low_01, low_blk_A_packs_0[pack_idx], low_blk_B_packs_1[pack_idx]);
                    iacc_high_01 = mul_sum_i8_pairs_acc_int32x8(iacc_high_01, high_blk_A_packs_0[pack_idx], high_blk_B_packs_1[pack_idx]);

                    // Second row of output
                    iacc_low_10 = mul_sum_i8_pairs_acc_int32x8(iacc_low_10, low_blk_A_packs_1[pack_idx], low_blk_B_packs_0[pack_idx]);
                    iacc_high_10 = mul_sum_i8_pairs_acc_int32x8(iacc_high_10, high_blk_A_packs_1[pack_idx], high_blk_B_packs_0[pack_idx]);
                    iacc_low_11 = mul_sum_i8_pairs_acc_int32x8(iacc_low_11, low_blk_A_packs_1[pack_idx], low_blk_B_packs_1[pack_idx]);
                    iacc_high_11 = mul_sum_i8_pairs_acc_int32x8(iacc_high_11, high_blk_A_packs_1[pack_idx], high_blk_B_packs_1[pack_idx]);
                }

                // Load scaling factors
                __m256 sB_low_0 = GGML_F32Cx8_LOAD(B_block_0.s_low);
                __m256 sB_high_0 = GGML_F32Cx8_LOAD(B_block_0.s_high);
                __m256 sB_low_1 = GGML_F32Cx8_LOAD(B_block_1.s_low);
                __m256 sB_high_1 = GGML_F32Cx8_LOAD(B_block_1.s_high);
                __m256 sA_low_0 = BROADCAST_FP16_FP32(A_ptr_0[blk_id].s_low);
                __m256 sA_high_0 = BROADCAST_FP16_FP32(A_ptr_0[blk_id].s_high);
                __m256 sA_low_1 = BROADCAST_FP16_FP32(A_ptr_1[blk_id].s_low);
                __m256 sA_high_1 = BROADCAST_FP16_FP32(A_ptr_1[blk_id].s_high);

                // Compute fused scaling factors
                __m256 s_low_00 = _mm256_mul_ps(sA_low_0, sB_low_0);
                __m256 s_high_00 = _mm256_mul_ps(sA_high_0, sB_high_0);
                __m256 s_low_01 = _mm256_mul_ps(sA_low_0, sB_low_1);
                __m256 s_high_01 = _mm256_mul_ps(sA_high_0, sB_high_1);
                __m256 s_low_10 = _mm256_mul_ps(sA_low_1, sB_low_0);
                __m256 s_high_10 = _mm256_mul_ps(sA_high_1, sB_high_0);
                __m256 s_low_11 = _mm256_mul_ps(sA_low_1, sB_low_1);
                __m256 s_high_11 = _mm256_mul_ps(sA_high_1, sB_high_1);

                // Accumulate results
                acc_row_00 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low_00), s_low_00, acc_row_00);
                acc_row_00 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high_00), s_high_00, acc_row_00);
                acc_row_01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low_01), s_low_01, acc_row_01);
                acc_row_01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high_01), s_high_01, acc_row_01);
                acc_row_10 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low_10), s_low_10, acc_row_10);
                acc_row_10 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high_10), s_high_10, acc_row_10);
                acc_row_11 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low_11), s_low_11, acc_row_11);
                acc_row_11 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high_11), s_high_11, acc_row_11);
            }

            // Store results
            switch (valid_i) {
                case 2:
                    _mm256_storeu_ps(C(i, j * 8, N), acc_row_00);
                    _mm256_storeu_ps(C(i, (j + 1) * 8, N), acc_row_01);
                    _mm256_storeu_ps(C(i + 1, j * 8, N), acc_row_10);
                    _mm256_storeu_ps(C(i + 1, (j + 1) * 8, N), acc_row_11);
                    break;
                case 1:
                    _mm256_storeu_ps(C(i, j * 8, N), acc_row_00);
                    _mm256_storeu_ps(C(i, (j + 1) * 8, N), acc_row_01);
            }
        }
    }
}

void gemm_repack_A81W41_fp32(
    void * A_repack,
    void * B_repack,
    float* C,
    const int M, const int N, const int K
)
{
    struct q4_repack_2x8_fp32 * B_ptr_start = (struct q4_repack_2x8_fp32 *) B_repack;
    struct q8_repack_1x2_fp32 * A_ptr_start = (struct q8_repack_1x2_fp32 *) A_repack;
    // how many q4_repack and q8_repack blocks along the N dimension
    int nb_k = K / (2 * Q_BLK_SIZE);
    // how many q4_repack blocks along the K dimension
    int nb_n = N / 8; 
    // how many q8_repack blocks along the M dimension
    int nb_m = M; // M is a multiple of 6 for this implementation
    PRAGMA_OMP_PARALLEL_FOR
        for(int j = 0; j < nb_n; j++)
    {
    for (int i = 0; i < nb_m; i += 6)
        {
            int valid_rows = min(6, nb_m - i);
            struct q8_repack_1x2_fp32 * A_ptr = A_ptr_start + i * nb_k;
            struct q8_repack_1x2_fp32 * A_ptr_next = A_ptr_start + (i + 1) * nb_k;
            struct q8_repack_1x2_fp32 * A_ptr_next2 = A_ptr_start + (i + 2) * nb_k;
            struct q8_repack_1x2_fp32 * A_ptr_next3 = A_ptr_start + (i + 3) * nb_k;
            struct q8_repack_1x2_fp32 * A_ptr_next4 = A_ptr_start + (i + 4) * nb_k;
            struct q8_repack_1x2_fp32 * A_ptr_next5 = A_ptr_start + (i + 5) * nb_k;
            struct q4_repack_2x8_fp32 * B_ptr = B_ptr_start + j * nb_k;
            
            __m256 acc_row0 = _mm256_setzero_ps();
            __m256 acc_row1 = _mm256_setzero_ps();
            __m256 acc_row2 = _mm256_setzero_ps();
            __m256 acc_row3 = _mm256_setzero_ps();
            __m256 acc_row4 = _mm256_setzero_ps();
            __m256 acc_row5 = _mm256_setzero_ps();
            
            for (int blk_id = 0; blk_id < nb_k; blk_id++)
            {
                struct q4_repack_2x8_fp32 B_block =  B_ptr[blk_id];
                // each pack contains 4 elements in the quantization block
                __m256i low_blk_B_packs[8];
                __m256i high_blk_B_packs[8];
                for (int pack_idx = 0; pack_idx < 8; pack_idx+=1)
                {
                    __m256i B_coupled_pack = _mm256_loadu_si256((__m256i *) (B_block.q_coupled + pack_idx * 32));
                    __m256i low_blk_B_pack = _mm256_and_si256(B_coupled_pack, _mm256_set1_epi8(0x0f));
                    __m256i high_blk_B_pack = _mm256_and_si256(_mm256_srli_epi16(B_coupled_pack, 4), _mm256_set1_epi8(0x0f));
                    low_blk_B_packs[pack_idx] = low_blk_B_pack;
                    high_blk_B_packs[pack_idx] = high_blk_B_pack;
                }
                
                // Process six rows of A simultaneously
                __m256i low_blk_A_packs[8];
                __m256i high_blk_A_packs[8];
                __m256i low_blk_A_packs_next[8];
                __m256i high_blk_A_packs_next[8];
                __m256i low_blk_A_packs_next2[8];
                __m256i high_blk_A_packs_next2[8];
                __m256i low_blk_A_packs_next3[8];
                __m256i high_blk_A_packs_next3[8];
                __m256i low_blk_A_packs_next4[8];
                __m256i high_blk_A_packs_next4[8];
                __m256i low_blk_A_packs_next5[8];
                __m256i high_blk_A_packs_next5[8];
                
                for (int pack_idx = 0; pack_idx < 8; pack_idx+=1)
                {
                    low_blk_A_packs[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr[blk_id].q_low) )[pack_idx]);
                    high_blk_A_packs[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr[blk_id].q_high) )[pack_idx]);
                    low_blk_A_packs_next[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr_next[blk_id].q_low) )[pack_idx]);
                    high_blk_A_packs_next[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr_next[blk_id].q_high) )[pack_idx]);
                    low_blk_A_packs_next2[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr_next2[blk_id].q_low) )[pack_idx]);
                    high_blk_A_packs_next2[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr_next2[blk_id].q_high) )[pack_idx]);
                    low_blk_A_packs_next3[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr_next3[blk_id].q_low) )[pack_idx]);
                    high_blk_A_packs_next3[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr_next3[blk_id].q_high) )[pack_idx]);
                    low_blk_A_packs_next4[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr_next4[blk_id].q_low) )[pack_idx]);
                    high_blk_A_packs_next4[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr_next4[blk_id].q_high) )[pack_idx]);
                    low_blk_A_packs_next5[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr_next5[blk_id].q_low) )[pack_idx]);
                    high_blk_A_packs_next5[pack_idx] = _mm256_set1_epi32(( (int32_t *)(A_ptr_next5[blk_id].q_high) )[pack_idx]);
                }
                
                __m256i iacc_low0 = _mm256_setzero_si256();
                __m256i iacc_high0 = _mm256_setzero_si256();
                __m256i iacc_low1 = _mm256_setzero_si256();
                __m256i iacc_high1 = _mm256_setzero_si256();
                __m256i iacc_low2 = _mm256_setzero_si256();
                __m256i iacc_high2 = _mm256_setzero_si256();
                __m256i iacc_low3 = _mm256_setzero_si256();
                __m256i iacc_high3 = _mm256_setzero_si256();
                __m256i iacc_low4 = _mm256_setzero_si256();
                __m256i iacc_high4 = _mm256_setzero_si256();
                __m256i iacc_low5 = _mm256_setzero_si256();
                __m256i iacc_high5 = _mm256_setzero_si256();
                
                for (int pack_idx = 0; pack_idx < 8; pack_idx+=1)
                {
                    iacc_low0  = mul_sum_us8_pairs_acc_int32x8 ( iacc_low0 , low_blk_B_packs  [ pack_idx ] , low_blk_A_packs  [ pack_idx ]); 
                    iacc_high0 = mul_sum_us8_pairs_acc_int32x8 ( iacc_high0, high_blk_B_packs [ pack_idx ] , high_blk_A_packs [ pack_idx ]); 
                    iacc_low1  = mul_sum_us8_pairs_acc_int32x8 ( iacc_low1 , low_blk_B_packs  [ pack_idx ] , low_blk_A_packs_next  [ pack_idx ]); 
                    iacc_high1 = mul_sum_us8_pairs_acc_int32x8 ( iacc_high1, high_blk_B_packs [ pack_idx ] , high_blk_A_packs_next [ pack_idx ]); 
                    iacc_low2  = mul_sum_us8_pairs_acc_int32x8 ( iacc_low2 , low_blk_B_packs  [ pack_idx ] , low_blk_A_packs_next2  [ pack_idx ]); 
                    iacc_high2 = mul_sum_us8_pairs_acc_int32x8 ( iacc_high2, high_blk_B_packs [ pack_idx ] , high_blk_A_packs_next2 [ pack_idx ]); 
                    iacc_low3  = mul_sum_us8_pairs_acc_int32x8 ( iacc_low3 , low_blk_B_packs  [ pack_idx ] , low_blk_A_packs_next3  [ pack_idx ]); 
                    iacc_high3 = mul_sum_us8_pairs_acc_int32x8 ( iacc_high3, high_blk_B_packs [ pack_idx ] , high_blk_A_packs_next3 [ pack_idx ]); 
                    iacc_low4  = mul_sum_us8_pairs_acc_int32x8 ( iacc_low4 , low_blk_B_packs  [ pack_idx ] , low_blk_A_packs_next4  [ pack_idx ]); 
                    iacc_high4 = mul_sum_us8_pairs_acc_int32x8 ( iacc_high4, high_blk_B_packs [ pack_idx ] , high_blk_A_packs_next4 [ pack_idx ]); 
                    iacc_low5  = mul_sum_us8_pairs_acc_int32x8 ( iacc_low5 , low_blk_B_packs  [ pack_idx ] , low_blk_A_packs_next5  [ pack_idx ]); 
                    iacc_high5 = mul_sum_us8_pairs_acc_int32x8 ( iacc_high5, high_blk_B_packs [ pack_idx ] , high_blk_A_packs_next5 [ pack_idx ]); 
                }
                
                // load scaleing factors
                __m256 sB_low = _mm256_loadu_ps(B_block.s_low);
                __m256 sB_high = _mm256_loadu_ps(B_block.s_high);
                __m256 minB_low = _mm256_loadu_ps(B_block.min_low);
                __m256 minB_high = _mm256_loadu_ps(B_block.min_high);

                __m256 sA_low = _mm256_set1_ps(A_ptr[blk_id].s_low);
                __m256 sA_high = _mm256_set1_ps(A_ptr[blk_id].s_high);
                __m256 scaled_sum_low = _mm256_set1_ps(A_ptr[blk_id].scaled_sum_low);
                __m256 scaled_sum_high = _mm256_set1_ps(A_ptr[blk_id].scaled_sum_high);

                __m256 sA_low_next = _mm256_set1_ps(A_ptr_next[blk_id].s_low);
                __m256 sA_high_next = _mm256_set1_ps(A_ptr_next[blk_id].s_high);
                __m256 scaled_sum_low_next = _mm256_set1_ps(A_ptr_next[blk_id].scaled_sum_low);
                __m256 scaled_sum_high_next = _mm256_set1_ps(A_ptr_next[blk_id].scaled_sum_high);

                __m256 sA_low_next2 = _mm256_set1_ps(A_ptr_next2[blk_id].s_low);
                __m256 sA_high_next2 = _mm256_set1_ps(A_ptr_next2[blk_id].s_high);
                __m256 scaled_sum_low_next2 = _mm256_set1_ps(A_ptr_next2[blk_id].scaled_sum_low);
                __m256 scaled_sum_high_next2 = _mm256_set1_ps(A_ptr_next2[blk_id].scaled_sum_high);

                __m256 sA_low_next3 = _mm256_set1_ps(A_ptr_next3[blk_id].s_low);
                __m256 sA_high_next3 = _mm256_set1_ps(A_ptr_next3[blk_id].s_high);
                __m256 scaled_sum_low_next3 = _mm256_set1_ps(A_ptr_next3[blk_id].scaled_sum_low);
                __m256 scaled_sum_high_next3 = _mm256_set1_ps(A_ptr_next3[blk_id].scaled_sum_high);

                __m256 sA_low_next4 = _mm256_set1_ps(A_ptr_next4[blk_id].s_low);
                __m256 sA_high_next4 = _mm256_set1_ps(A_ptr_next4[blk_id].s_high);
                __m256 scaled_sum_low_next4 = _mm256_set1_ps(A_ptr_next4[blk_id].scaled_sum_low);
                __m256 scaled_sum_high_next4 = _mm256_set1_ps(A_ptr_next4[blk_id].scaled_sum_high);

                __m256 sA_low_next5 = _mm256_set1_ps(A_ptr_next5[blk_id].s_low);
                __m256 sA_high_next5 = _mm256_set1_ps(A_ptr_next5[blk_id].s_high);
                __m256 scaled_sum_low_next5 = _mm256_set1_ps(A_ptr_next5[blk_id].scaled_sum_low);
                __m256 scaled_sum_high_next5 = _mm256_set1_ps(A_ptr_next5[blk_id].scaled_sum_high);

                __m256 s_low = _mm256_mul_ps(sA_low, sB_low);
                __m256 s_high = _mm256_mul_ps(sA_high, sB_high);
                __m256 s_low_next = _mm256_mul_ps(sA_low_next, sB_low);
                __m256 s_high_next = _mm256_mul_ps(sA_high_next, sB_high);
                __m256 s_low_next2 = _mm256_mul_ps(sA_low_next2, sB_low);
                __m256 s_high_next2 = _mm256_mul_ps(sA_high_next2, sB_high);
                __m256 s_low_next3 = _mm256_mul_ps(sA_low_next3, sB_low);
                __m256 s_high_next3 = _mm256_mul_ps(sA_high_next3, sB_high);
                __m256 s_low_next4 = _mm256_mul_ps(sA_low_next4, sB_low);
                __m256 s_high_next4 = _mm256_mul_ps(sA_high_next4, sB_high);
                __m256 s_low_next5 = _mm256_mul_ps(sA_low_next5, sB_low);
                __m256 s_high_next5 = _mm256_mul_ps(sA_high_next5, sB_high);

                // First row
                acc_row0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low0), s_low, acc_row0);
                acc_row0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high0), s_high, acc_row0);
                acc_row0 = _mm256_fmadd_ps(minB_low, scaled_sum_low, acc_row0);
                acc_row0 = _mm256_fmadd_ps(minB_high, scaled_sum_high, acc_row0);

                // Second row
                acc_row1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low1), s_low_next, acc_row1);
                acc_row1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high1), s_high_next, acc_row1);
                acc_row1 = _mm256_fmadd_ps(minB_low, scaled_sum_low_next, acc_row1);
                acc_row1 = _mm256_fmadd_ps(minB_high, scaled_sum_high_next, acc_row1);

                // Third row
                acc_row2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low2), s_low_next2, acc_row2);
                acc_row2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high2), s_high_next2, acc_row2);
                acc_row2 = _mm256_fmadd_ps(minB_low, scaled_sum_low_next2, acc_row2);
                acc_row2 = _mm256_fmadd_ps(minB_high, scaled_sum_high_next2, acc_row2);

                // Fourth row
                acc_row3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low3), s_low_next3, acc_row3);
                acc_row3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high3), s_high_next3, acc_row3);
                acc_row3 = _mm256_fmadd_ps(minB_low, scaled_sum_low_next3, acc_row3);
                acc_row3 = _mm256_fmadd_ps(minB_high, scaled_sum_high_next3, acc_row3);

                // Fifth row
                acc_row4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low4), s_low_next4, acc_row4);
                acc_row4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high4), s_high_next4, acc_row4);
                acc_row4 = _mm256_fmadd_ps(minB_low, scaled_sum_low_next4, acc_row4);
                acc_row4 = _mm256_fmadd_ps(minB_high, scaled_sum_high_next4, acc_row4);

                // Sixth row
                acc_row5 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_low5), s_low_next5, acc_row5);
                acc_row5 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_high5), s_high_next5, acc_row5);
                acc_row5 = _mm256_fmadd_ps(minB_low, scaled_sum_low_next5, acc_row5);
                acc_row5 = _mm256_fmadd_ps(minB_high, scaled_sum_high_next5, acc_row5);
            }
        switch (valid_rows) {
            case 6:
                _mm256_storeu_ps(C(i, j * 8, N), acc_row0);
                _mm256_storeu_ps(C(i + 1, j * 8, N), acc_row1);
                _mm256_storeu_ps(C(i + 2, j * 8, N), acc_row2);
                _mm256_storeu_ps(C(i + 3, j * 8, N), acc_row3);
                _mm256_storeu_ps(C(i + 4, j * 8, N), acc_row4);
                _mm256_storeu_ps(C(i + 5, j * 8, N), acc_row5);
                break;
            case 5:
                _mm256_storeu_ps(C(i, j * 8, N), acc_row0);
                _mm256_storeu_ps(C(i + 1, j * 8, N), acc_row1);
                _mm256_storeu_ps(C(i + 2, j * 8, N), acc_row2);
                _mm256_storeu_ps(C(i + 3, j * 8, N), acc_row3);
                _mm256_storeu_ps(C(i + 4, j * 8, N), acc_row4);
                break;
            case 4:
                _mm256_storeu_ps(C(i, j * 8, N), acc_row0);
                _mm256_storeu_ps(C(i + 1, j * 8, N), acc_row1);
                _mm256_storeu_ps(C(i + 2, j * 8, N), acc_row2);
                _mm256_storeu_ps(C(i + 3, j * 8, N), acc_row3);
                break;
            case 3:
                _mm256_storeu_ps(C(i, j * 8, N), acc_row0);
                _mm256_storeu_ps(C(i + 1, j * 8, N), acc_row1);
                _mm256_storeu_ps(C(i + 2, j * 8, N), acc_row2);
                break;
            case 2:
                _mm256_storeu_ps(C(i, j * 8, N), acc_row0);
                _mm256_storeu_ps(C(i + 1, j * 8, N), acc_row1);
                break;
            case 1:
                _mm256_storeu_ps(C(i, j * 8, N), acc_row0);
        }
        }
    }


}

void quantize_row_q8_0_repack(const float * x, void * vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    struct q8_repack_1x2_fp16 * y = (struct q8_repack_1x2_fp16 *) vy;
    for (int i = 0; i < nb/2; i++) {
        // Load elements into 8 AVX vectors (4 for each block)
        __m256 v0_low = _mm256_loadu_ps(x);
        __m256 v1_low = _mm256_loadu_ps(x + 8);
        __m256 v2_low = _mm256_loadu_ps(x + 16);
        __m256 v3_low = _mm256_loadu_ps(x + 24);
        __m256 v0_high = _mm256_loadu_ps(x + 32);
        __m256 v1_high = _mm256_loadu_ps(x + 40);
        __m256 v2_high = _mm256_loadu_ps(x + 48);
        __m256 v3_high = _mm256_loadu_ps(x + 56);
        x += 64;

        // Compute max(abs(e)) for each block
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        
        // For low block
        __m256 maxAbs_low = _mm256_andnot_ps(signBit, v0_low);
        maxAbs_low = _mm256_max_ps(maxAbs_low, _mm256_andnot_ps(signBit, v1_low));
        maxAbs_low = _mm256_max_ps(maxAbs_low, _mm256_andnot_ps(signBit, v2_low));
        maxAbs_low = _mm256_max_ps(maxAbs_low, _mm256_andnot_ps(signBit, v3_low));

        // For high block
        __m256 maxAbs_high = _mm256_andnot_ps(signBit, v0_high);
        maxAbs_high = _mm256_max_ps(maxAbs_high, _mm256_andnot_ps(signBit, v1_high));
        maxAbs_high = _mm256_max_ps(maxAbs_high, _mm256_andnot_ps(signBit, v2_high));
        maxAbs_high = _mm256_max_ps(maxAbs_high, _mm256_andnot_ps(signBit, v3_high));

        // Get max values
        __m128 max4_low = _mm_max_ps(_mm256_extractf128_ps(maxAbs_low, 1), _mm256_castps256_ps128(maxAbs_low));
        max4_low = _mm_max_ps(max4_low, _mm_movehl_ps(max4_low, max4_low));
        max4_low = _mm_max_ss(max4_low, _mm_movehdup_ps(max4_low));
        const float maxScalar_low = _mm_cvtss_f32(max4_low);

        __m128 max4_high = _mm_max_ps(_mm256_extractf128_ps(maxAbs_high, 1), _mm256_castps256_ps128(maxAbs_high));
        max4_high = _mm_max_ps(max4_high, _mm_movehl_ps(max4_high, max4_high));
        max4_high = _mm_max_ss(max4_high, _mm_movehdup_ps(max4_high));
        const float maxScalar_high = _mm_cvtss_f32(max4_high);

        // Quantize these floats
        const float d_low = maxScalar_low / 127.f;
        const float d_high = maxScalar_high / 127.f;
        y[i].s_low = GGML_FP32_TO_FP16(d_low);
        y[i].s_high = GGML_FP32_TO_FP16(d_high);
        
        const float id_low = (maxScalar_low != 0.0f) ? 127.f / maxScalar_low : 0.0f;
        const float id_high = (maxScalar_high != 0.0f) ? 127.f / maxScalar_high : 0.0f;
        const __m256 mul_low = _mm256_set1_ps(id_low);
        const __m256 mul_high = _mm256_set1_ps(id_high);

        // Apply the multipliers
        v0_low = _mm256_mul_ps(v0_low, mul_low);
        v1_low = _mm256_mul_ps(v1_low, mul_low);
        v2_low = _mm256_mul_ps(v2_low, mul_low);
        v3_low = _mm256_mul_ps(v3_low, mul_low);
        
        v0_high = _mm256_mul_ps(v0_high, mul_high);
        v1_high = _mm256_mul_ps(v1_high, mul_high);
        v2_high = _mm256_mul_ps(v2_high, mul_high);
        v3_high = _mm256_mul_ps(v3_high, mul_high);

        // Round to nearest integer
        v0_low = _mm256_round_ps(v0_low, _MM_ROUND_NEAREST);
        v1_low = _mm256_round_ps(v1_low, _MM_ROUND_NEAREST);
        v2_low = _mm256_round_ps(v2_low, _MM_ROUND_NEAREST);
        v3_low = _mm256_round_ps(v3_low, _MM_ROUND_NEAREST);
        
        v0_high = _mm256_round_ps(v0_high, _MM_ROUND_NEAREST);
        v1_high = _mm256_round_ps(v1_high, _MM_ROUND_NEAREST);
        v2_high = _mm256_round_ps(v2_high, _MM_ROUND_NEAREST);
        v3_high = _mm256_round_ps(v3_high, _MM_ROUND_NEAREST);

        // Convert floats to integers
        __m256i i0_low = _mm256_cvtps_epi32(v0_low);
        __m256i i1_low = _mm256_cvtps_epi32(v1_low);
        __m256i i2_low = _mm256_cvtps_epi32(v2_low);
        __m256i i3_low = _mm256_cvtps_epi32(v3_low);
        
        __m256i i0_high = _mm256_cvtps_epi32(v0_high);
        __m256i i1_high = _mm256_cvtps_epi32(v1_high);
        __m256i i2_high = _mm256_cvtps_epi32(v2_high);
        __m256i i3_high = _mm256_cvtps_epi32(v3_high);

        // Convert int32 to int16
        i0_low = _mm256_packs_epi32(i0_low, i1_low);
        i2_low = _mm256_packs_epi32(i2_low, i3_low);
        i0_high = _mm256_packs_epi32(i0_high, i1_high);
        i2_high = _mm256_packs_epi32(i2_high, i3_high);

        // Convert int16 to int8
        i0_low = _mm256_packs_epi16(i0_low, i2_low);
        i0_high = _mm256_packs_epi16(i0_high, i2_high);

        // Fix the order
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0_low = _mm256_permutevar8x32_epi32(i0_low, perm);
        i0_high = _mm256_permutevar8x32_epi32(i0_high, perm);

        // Store results
        _mm256_storeu_si256((__m256i *)y[i].q_low, i0_low);
        _mm256_storeu_si256((__m256i *)y[i].q_high, i0_high);
    }
}

void quantize_row_q8_1_repack_fp16(const float * x, void * vy, int64_t k) {
    assert(k % (2 * QK8_1) == 0);  // Must be multiple of 64 (2 blocks of 32)
    const int nb = k / (2 * QK8_1);  // Number of q80_repack_1x2 blocks

    struct q8_repack_1x2_fp16 * GGML_RESTRICT y = (struct q8_repack_1x2_fp16 *) vy;
    for (int i = 0; i < nb; i++) {
        // Load elements into 8 AVX vectors (64 elements total)
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        __m256 v4 = _mm256_loadu_ps(x + 32);
        __m256 v5 = _mm256_loadu_ps(x + 40);
        __m256 v6 = _mm256_loadu_ps(x + 48);
        __m256 v7 = _mm256_loadu_ps(x + 56);
        x += 64;

        // Compute max(abs(e)) for first block (first 32 elements)
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs1 = _mm256_andnot_ps(signBit, v0);
        maxAbs1 = _mm256_max_ps(maxAbs1, _mm256_andnot_ps(signBit, v1));
        maxAbs1 = _mm256_max_ps(maxAbs1, _mm256_andnot_ps(signBit, v2));
        maxAbs1 = _mm256_max_ps(maxAbs1, _mm256_andnot_ps(signBit, v3));

        // Compute max(abs(e)) for second block (next 32 elements)
        __m256 maxAbs2 = _mm256_andnot_ps(signBit, v4);
        maxAbs2 = _mm256_max_ps(maxAbs2, _mm256_andnot_ps(signBit, v5));
        maxAbs2 = _mm256_max_ps(maxAbs2, _mm256_andnot_ps(signBit, v6));
        maxAbs2 = _mm256_max_ps(maxAbs2, _mm256_andnot_ps(signBit, v7));

        // Get max values for both blocks
        __m128 max4_1 = _mm_max_ps(_mm256_extractf128_ps(maxAbs1, 1), _mm256_castps256_ps128(maxAbs1));
        max4_1 = _mm_max_ps(max4_1, _mm_movehl_ps(max4_1, max4_1));
        max4_1 = _mm_max_ss(max4_1, _mm_movehdup_ps(max4_1));
        const float max_scalar1 = _mm_cvtss_f32(max4_1);

        __m128 max4_2 = _mm_max_ps(_mm256_extractf128_ps(maxAbs2, 1), _mm256_castps256_ps128(maxAbs2));
        max4_2 = _mm_max_ps(max4_2, _mm_movehl_ps(max4_2, max4_2));
        max4_2 = _mm_max_ss(max4_2, _mm_movehdup_ps(max4_2));
        const float max_scalar2 = _mm_cvtss_f32(max4_2);

        // Quantize first block
        const float d1 = max_scalar1 / 127.f;
        y[i].s_low = GGML_FP32_TO_FP16(d1);
        const float id1 = (max_scalar1 != 0.0f) ? 127.f / max_scalar1 : 0.0f;
        const __m256 mul1 = _mm256_set1_ps(id1);

        // Quantize second block
        const float d2 = max_scalar2 / 127.f;
        y[i].s_high = GGML_FP32_TO_FP16(d2);
        const float id2 = (max_scalar2 != 0.0f) ? 127.f / max_scalar2 : 0.0f;
        const __m256 mul2 = _mm256_set1_ps(id2);

        // Apply multipliers
        v0 = _mm256_mul_ps(v0, mul1);
        v1 = _mm256_mul_ps(v1, mul1);
        v2 = _mm256_mul_ps(v2, mul1);
        v3 = _mm256_mul_ps(v3, mul1);
        v4 = _mm256_mul_ps(v4, mul2);
        v5 = _mm256_mul_ps(v5, mul2);
        v6 = _mm256_mul_ps(v6, mul2);
        v7 = _mm256_mul_ps(v7, mul2);

        // Round to nearest integer
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);
        v4 = _mm256_round_ps(v4, _MM_ROUND_NEAREST);
        v5 = _mm256_round_ps(v5, _MM_ROUND_NEAREST);
        v6 = _mm256_round_ps(v6, _MM_ROUND_NEAREST);
        v7 = _mm256_round_ps(v7, _MM_ROUND_NEAREST);

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        __m256i i4 = _mm256_cvtps_epi32(v4);
        __m256i i5 = _mm256_cvtps_epi32(v5);
        __m256i i6 = _mm256_cvtps_epi32(v6);
        __m256i i7 = _mm256_cvtps_epi32(v7);

        // Compute scaled sums for both blocks
        float scaled_sum_low = ( d1 * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))) );
        y[i].scaled_sum_low = GGML_FP32_TO_FP16(scaled_sum_low);
        float scaled_sum_high = ( d2 * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i4, i5), _mm256_add_epi32(i6, i7))) );
        y[i].scaled_sum_high = GGML_FP32_TO_FP16(scaled_sum_high);

        // Convert int32 to int16 for first block
        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);

        // Convert int32 to int16 for second block
        i4 = _mm256_packs_epi32(i4, i5);
        i6 = _mm256_packs_epi32(i6, i7);
        i4 = _mm256_packs_epi16(i4, i6);

        // Fix order for first block
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        i4 = _mm256_permutevar8x32_epi32(i4, perm);

        // Store results
        _mm256_storeu_si256((__m256i *)y[i].q_low, i0);
        _mm256_storeu_si256((__m256i *)y[i].q_high, i4);
    }
}

void quantize_row_q8_1_repack_fp32(const float * x, void * vy, int64_t k) {
    assert(k % (2 * QK8_1) == 0);  // Must be multiple of 64 (2 blocks of 32)
    const int nb = k / (2 * QK8_1);  // Number of q80_repack_1x2 blocks

    struct q8_repack_1x2_fp32 * GGML_RESTRICT y = (struct q8_repack_1x2_fp32 *) vy;
    for (int i = 0; i < nb; i++) {
        // Load elements into 8 AVX vectors (64 elements total)
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        __m256 v4 = _mm256_loadu_ps(x + 32);
        __m256 v5 = _mm256_loadu_ps(x + 40);
        __m256 v6 = _mm256_loadu_ps(x + 48);
        __m256 v7 = _mm256_loadu_ps(x + 56);
        x += 64;

        // Compute max(abs(e)) for first block (first 32 elements)
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs1 = _mm256_andnot_ps(signBit, v0);
        maxAbs1 = _mm256_max_ps(maxAbs1, _mm256_andnot_ps(signBit, v1));
        maxAbs1 = _mm256_max_ps(maxAbs1, _mm256_andnot_ps(signBit, v2));
        maxAbs1 = _mm256_max_ps(maxAbs1, _mm256_andnot_ps(signBit, v3));

        // Compute max(abs(e)) for second block (next 32 elements)
        __m256 maxAbs2 = _mm256_andnot_ps(signBit, v4);
        maxAbs2 = _mm256_max_ps(maxAbs2, _mm256_andnot_ps(signBit, v5));
        maxAbs2 = _mm256_max_ps(maxAbs2, _mm256_andnot_ps(signBit, v6));
        maxAbs2 = _mm256_max_ps(maxAbs2, _mm256_andnot_ps(signBit, v7));

        // Get max values for both blocks
        __m128 max4_1 = _mm_max_ps(_mm256_extractf128_ps(maxAbs1, 1), _mm256_castps256_ps128(maxAbs1));
        max4_1 = _mm_max_ps(max4_1, _mm_movehl_ps(max4_1, max4_1));
        max4_1 = _mm_max_ss(max4_1, _mm_movehdup_ps(max4_1));
        const float max_scalar1 = _mm_cvtss_f32(max4_1);

        __m128 max4_2 = _mm_max_ps(_mm256_extractf128_ps(maxAbs2, 1), _mm256_castps256_ps128(maxAbs2));
        max4_2 = _mm_max_ps(max4_2, _mm_movehl_ps(max4_2, max4_2));
        max4_2 = _mm_max_ss(max4_2, _mm_movehdup_ps(max4_2));
        const float max_scalar2 = _mm_cvtss_f32(max4_2);

        // Quantize first block
        const float d1 = max_scalar1 / 127.f;
        y[i].s_low = d1;
        const float id1 = (max_scalar1 != 0.0f) ? 127.f / max_scalar1 : 0.0f;
        const __m256 mul1 = _mm256_set1_ps(id1);

        // Quantize second block
        const float d2 = max_scalar2 / 127.f;
        y[i].s_high = d2;
        const float id2 = (max_scalar2 != 0.0f) ? 127.f / max_scalar2 : 0.0f;
        const __m256 mul2 = _mm256_set1_ps(id2);

        // Apply multipliers
        v0 = _mm256_mul_ps(v0, mul1);
        v1 = _mm256_mul_ps(v1, mul1);
        v2 = _mm256_mul_ps(v2, mul1);
        v3 = _mm256_mul_ps(v3, mul1);
        v4 = _mm256_mul_ps(v4, mul2);
        v5 = _mm256_mul_ps(v5, mul2);
        v6 = _mm256_mul_ps(v6, mul2);
        v7 = _mm256_mul_ps(v7, mul2);

        // Round to nearest integer
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);
        v4 = _mm256_round_ps(v4, _MM_ROUND_NEAREST);
        v5 = _mm256_round_ps(v5, _MM_ROUND_NEAREST);
        v6 = _mm256_round_ps(v6, _MM_ROUND_NEAREST);
        v7 = _mm256_round_ps(v7, _MM_ROUND_NEAREST);

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        __m256i i4 = _mm256_cvtps_epi32(v4);
        __m256i i5 = _mm256_cvtps_epi32(v5);
        __m256i i6 = _mm256_cvtps_epi32(v6);
        __m256i i7 = _mm256_cvtps_epi32(v7);

        // Compute scaled sums for both blocks
        float scaled_sum_low = ( d1 * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))) );
        y[i].scaled_sum_low = scaled_sum_low;
        float scaled_sum_high = ( d2 * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i4, i5), _mm256_add_epi32(i6, i7))) );
        y[i].scaled_sum_high = scaled_sum_high;

        // Convert int32 to int16 for first block
        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);

        // Convert int32 to int16 for second block
        i4 = _mm256_packs_epi32(i4, i5);
        i6 = _mm256_packs_epi32(i6, i7);
        i4 = _mm256_packs_epi16(i4, i6);

        // Fix order for first block
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        i4 = _mm256_permutevar8x32_epi32(i4, perm);

        // Store results
        _mm256_storeu_si256((__m256i *)y[i].q_low, i0);
        _mm256_storeu_si256((__m256i *)y[i].q_high, i4);
    }
}