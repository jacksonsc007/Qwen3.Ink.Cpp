#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "matmul.h"
#include "common.h"


namespace matmul {


void MatmulOperator::mat_mul_loop_unrolling4x4_secondInnermost_tiling2level_fp32(struct matmul_params *params) 
{
    
    // printf("\e[31m[INFO]\e[m hi\n");
    // A: activation        (m, k) 
    // B: transposed weight (k, n)
    // C: result            (m, n)
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    // printf("[INFO] : m=%d, n=%d, k=%d\n", m, n, k);
    int tile_size = 64;

    // zero out matrix C
    for (int i = 0; i < m * n; i++)
        C->data_ptr[i] = 0.0;

// exit(1);
for (int tile_start_col = 0; tile_start_col < n; tile_start_col += tile_size) // tile_start_col: the start column idx for a tile in matrix B
for (int tile_start_row = 0; tile_start_row < k; tile_start_row += tile_size) // tile_start_col: the start row idx for a tile in matrix B
    // tile_A: (m, tile_size) tile_B: (tile_size, tile_size)
    for (int row = 0; row < m; row++)
        for (int col = 0; col < tile_size; col+=4)
        {
            float sum0 = 0.0;
            float sum1 = 0.0;
            float sum2 = 0.0;
            float sum3 = 0.0;
            for (int ch = 0; ch < tile_size; ch++)
            {
                // A (row, tile_start_row + ch); B (tile_start_row + ch, tile_start_colum + col)
                // C (row, tile_start_colum + col)
                sum0 += A->data_ptr[row * k + ch + tile_start_row] * B->data_ptr[(col +     tile_start_col) * k + ch + tile_start_row];
                sum1 += A->data_ptr[row * k + ch + tile_start_row] * B->data_ptr[(col + 1 + tile_start_col) * k + ch + tile_start_row];
                sum2 += A->data_ptr[row * k + ch + tile_start_row] * B->data_ptr[(col + 2 + tile_start_col) * k + ch + tile_start_row];
                sum3 += A->data_ptr[row * k + ch + tile_start_row] * B->data_ptr[(col + 3 + tile_start_col) * k + ch + tile_start_row];
            }
            C->data_ptr[row * n + col + tile_start_col] += sum0;
            C->data_ptr[row * n + col + 1 + tile_start_col] += sum1;
            C->data_ptr[row * n + col + 2 + tile_start_col] += sum2;
            C->data_ptr[row * n + col + 3 + tile_start_col] += sum3;
        }
}
void MatmulOperator::mat_mul_loop_unrolling4x4_secondInnermost_tiling_fp32(struct matmul_params *params) 
{
    
    // printf("\e[31m[INFO]\e[m Fp32 unrolling version\n");
    // A: activation        (m, k) 
    // B: transposed weight (k, n)
    // C: result            (m, n)
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    // printf("[INFO] : m=%d, n=%d, k=%d\n", m, n, k);
    int tile_size = 32;


// exit(1);
for (int tile_start_idx = 0; tile_start_idx < n; tile_start_idx += tile_size)
    for (int row = 0; row < m; row++)
        for (int col = 0; col < tile_size; col+=4)
        {
            float sum0 = 0.0;
            float sum1 = 0.0;
            float sum2 = 0.0;
            float sum3 = 0.0;
            for (int ch = 0; ch < k; ch++)
            {
                sum0 += A->data_ptr[row * k + ch] * B->data_ptr[(col +     tile_start_idx) * k + ch ];
                sum1 += A->data_ptr[row * k + ch] * B->data_ptr[(col + 1 + tile_start_idx) * k + ch ];
                sum2 += A->data_ptr[row * k + ch] * B->data_ptr[(col + 2 + tile_start_idx) * k + ch ];
                sum3 += A->data_ptr[row * k + ch] * B->data_ptr[(col + 3 + tile_start_idx) * k + ch ];
            }
            C->data_ptr[row * n + col + tile_start_idx] = sum0;
            C->data_ptr[row * n + col + 1 + tile_start_idx] = sum1;
            C->data_ptr[row * n + col + 2 + tile_start_idx] = sum2;
            C->data_ptr[row * n + col + 3 + tile_start_idx] = sum3;
        }
}
void MatmulOperator::mat_mul_loop_unrolling4x4_tiling_fp32(struct matmul_params *params) 
{
    
    // printf("\e[31m[INFO]\e[m Fp32 unrolling version\n");
    // A: activation        (m, k) 
    // B: transposed weight (k, n)
    // C: result            (m, n)
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    // printf("[INFO] matmul_loop_unrolling4x4_fp32: m=%d, n=%d, k=%d\n", m, n, k);
    int tile_size = 64;


// exit(1);
for (int tile_start_idx = 0; tile_start_idx < n; tile_start_idx += tile_size)
    for (int row = 0; row < m; row++)
        for (int col = 0; col < tile_size; col++)
        {
            float sum0 = 0.0;
            float sum1 = 0.0;
            float sum2 = 0.0;
            float sum3 = 0.0;
            int ch = 0;
            for (; ch < k - 4;)
            {
                sum0 += A->data_ptr[row * k + ch]     * B->data_ptr[(col + tile_start_idx) * k + ch   ];
                sum1 += A->data_ptr[row * k + ch + 1] * B->data_ptr[(col + tile_start_idx) * k + ch + 1];
                sum2 += A->data_ptr[row * k + ch + 2] * B->data_ptr[(col + tile_start_idx) * k + ch + 2];
                sum3 += A->data_ptr[row * k + ch + 3] * B->data_ptr[(col + tile_start_idx) * k + ch + 3];
                ch += 4;
            }
            for (; ch < k; ch++)
            {
                // printf("\e[31m[INFO]\e[m dealing with remaining channels...\n");
                sum0 += A->data_ptr[row * k + ch]     * B->data_ptr[(col + tile_start_idx) * k + ch];

            }
            C->data_ptr[row * n + col + tile_start_idx] = sum0 + sum1 + sum2 + sum3;
        }
}

}  // namespace matmul
