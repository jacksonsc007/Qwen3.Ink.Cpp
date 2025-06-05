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

void MatmulOperator::matMul_avx_tiling2level_fp32(struct matmul_params *params) {
    // printf("hi\n");
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    // zero out C
    for (int i = 0; i < m * n; i++)
        data_C[i] = 0;

        
    int tile_size = 32;
    // one level tiling on the column dimension of matric B of shape k, n
    for (int tile_start_col = 0; tile_start_col < n; tile_start_col += tile_size)
    for (int tile_start_row = 0; tile_start_row < k; tile_start_row += tile_size)
    // tile_of_A: (m, tile_size) tile_of_B: (tile_size, tile_size)
    for (int i = 0; i < m; i++) 
    {
        // size of tile_of_A: (1, tile_size) size of tile_of_B: (tile_size, tile_size)
        // data_C: from (i, tile_start_col) to (i, tile_start_col + tile_size)
        for (int j = 0; j < tile_size; j++) 
        {
            // 32 = 4 * 8
            __m256 acc_array[4];
            for (int r_id = 0; r_id < 4; r_id++)
            {
                acc_array[r_id] = _mm256_load_ps(data_C + i * k + tile_start_col + 8 * r_id);
            }
            __m256 b_packed_array[4];

            for (int ch = 0; ch < tile_size; ch++)
            {
                // A: (i, tile_start_row + ch); B: (tile_start_row + ch, tile_start_col + j)
                // C: (i, tile_start_col + j)
                // __m256 a_val = _mm256_loadu_ps(data_A + i * k                    + tile_start_row + ch);
                // __m256 b_val = _mm256_loadu_ps(data_B + (tile_start_col + j) * k + tile_start_row + ch);
                // acc_packed = _mm256_add_ps(acc_packed, _mm256_mul_ps(a_val, b_val));
                        
                // B: (tile_start_row + ch, tile_start_col + j)
                b_packed_array[0] = _mm256_loadu_ps(data_B + (tile_start_col) * k      + tile_start_row + ch);
                b_packed_array[1] = _mm256_loadu_ps(data_B + (tile_start_col + 8) * k  + tile_start_row + ch);
                b_packed_array[2] = _mm256_loadu_ps(data_B + (tile_start_col + 16) * k + tile_start_row + ch);
                b_packed_array[3] = _mm256_loadu_ps(data_B + (tile_start_col + 24) * k + tile_start_row + ch);
                
                __m256 a_packed = _mm256_broadcast_ss(data_A + i * k + tile_start_row + ch);
                acc_array[0] = _mm256_fmadd_ps(a_packed, b_packed_array[0], acc_array[0]);
                acc_array[1] = _mm256_fmadd_ps(a_packed, b_packed_array[1], acc_array[1]);
                acc_array[2] = _mm256_fmadd_ps(a_packed, b_packed_array[2], acc_array[2]);
                acc_array[3] = _mm256_fmadd_ps(a_packed, b_packed_array[3], acc_array[3]);
                
            }
            
            // C: (i, tile_start_col)
            for (int r_id = 0; r_id < 4; r_id++)
            {
                _mm256_store_ps(data_C + i * k + tile_start_col, acc_array[r_id]);
            }
        }
    }
}

// void MatmulOperator::matMul_avx_tiling2level_fp32(struct matmul_params *params) {
//     // printf("hi\n");
//     const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
//     float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

//     assert(A->column == B->row);
//     assert(C->row == A->row);
//     assert(C->column == B->column);
//     int m = A->row, n = B->column, k = A->column;

//     // zero out C
//     for (int i = 0; i < m * n; i++)
//         data_C[i] = 0;


//     int tile_size = 64;
//     // one level tiling on the column dimension of matric B of shape k, n
//     for (int tile_start_col = 0; tile_start_col < n; tile_start_col += tile_size)
//     for (int tile_start_row = 0; tile_start_row < k; tile_start_row += tile_size)
//     // tile_of_A: (m, tile_size) tile_of_B: (tile_size, tile_size)
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < tile_size; j++) {
//             __m256 acc_packed = _mm256_setzero_ps();

//             for (int ch = 0; ch < tile_size; ch += 8) {
//                 // A: (i, tile_start_row + ch); B: (tile_start_row + ch, tile_start_col + j)
//                 // C: (i, tile_start_col + j)
//                 __m256 a_val = _mm256_loadu_ps(data_A + i * k                    + tile_start_row + ch);
//                 __m256 b_val = _mm256_loadu_ps(data_B + (tile_start_col + j) * k + tile_start_row + ch);
//                 acc_packed = _mm256_add_ps(acc_packed, _mm256_mul_ps(a_val, b_val));
//             }
            
//             float * ptr0 = (float*) &acc_packed;
//             data_C[i * n + j + tile_start_col] += ptr0[0] + ptr0[1] + ptr0[2] + ptr0[3] + 
//                         ptr0[4] + ptr0[5] + ptr0[6] + ptr0[7];
//         }
//     }
// }
void MatmulOperator::matMul_avx_tiling_fp32(struct matmul_params *params) {
    // printf("hi\n");
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    int tile_size = 32;
    // one level tiling on the column dimension of matric B of shape k, n
    for (int tile_start_col = 0; tile_start_col < n; tile_start_col += tile_size)
    // A: (m, k) tile_of_B: (k, tile_size)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < tile_size; j++) {
            float acc = 0;
            __m256 acc_packed = _mm256_setzero_ps();
            int kk = 0;
            for (; kk <= k-8; kk+=8) {
                // A: (i, kk) B: (kk, j+tile_start_col)
                __m256 * a_ptr = (__m256 *) (data_A + (i                 ) * k + kk);
                __m256 * b_ptr = (__m256 *) (data_B + (j + tile_start_col) * k + kk);
                __m256 a_val = _mm256_loadu_ps((float *) a_ptr);
                __m256 b_val = _mm256_loadu_ps((float *) b_ptr);
                acc_packed = _mm256_add_ps(acc_packed, _mm256_mul_ps(a_val, b_val));
            }
            
            // NOTE: We annote this for the sake of performance, but we ensure that there is no remainder.
            // for (;kk < k; kk++)
            // {
            //     acc += data_A[i * k + kk] * data_B[j * k + kk];                
            // }

            float * ptr0 = (float*) &acc_packed;
            acc += ptr0[0] + ptr0[1] + ptr0[2] + ptr0[3] + 
                        ptr0[4] + ptr0[5] + ptr0[6] + ptr0[7];
            data_C[i * n + j + tile_start_col] = acc;
        }
    }
}


}  // namespace matmul
