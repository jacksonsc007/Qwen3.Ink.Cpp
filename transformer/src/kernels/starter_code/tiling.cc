#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "matmul.h"
#include "common.h"

namespace matmul {
void MatmulOperator::matMul_tiling2level_fp32(struct matmul_params *params) {
    /*
     *  C = A x B; We divide B from two dimension, and A from column dimension.
    */
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    // printf("hi\n");

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    //  1 x 4096, 4096 x 4096
    //  1 x 4096, 4096 x 151936
    int m = A->row, n = B->column, k = A->column; 
    int tile_size = 64;
    // int remain = n % tile_size;
    // if (remain != 0)
    // {
    //     printf("\e[31m[INFO]\e[m tiling size\n");
    //     throw("tiling size error");
    // }
    
    // zero the C
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            data_C[i * n + j] = 0;

    // NOTE: the presedence of tiling level matter!
    for (int tile_start_x = 0; tile_start_x < n; tile_start_x += tile_size)     // divide B in column dimension
    for (int tile_start_y = 0; tile_start_y < k; tile_start_y += tile_size) // divide B in row dimension, and A in column dimension
    for (int i = 0; i < m; i++) 
    for (int j = 0; j < tile_size; j++) 
    {
        float sum = 0;
        for (int kk = 0; kk < tile_size; kk++) 
        {
            // A(i, tile_start_y + kk)  B(tile_start_y + kk, tile_start_x + j)
            // C(i, tile_start_x + j)
             sum += data_A[i * k + tile_start_y + kk] * data_B[(tile_start_x + j) * k + kk + tile_start_y];
        }
        data_C[i * n + tile_start_x + j] += sum;

    }
        
}

void MatmulOperator::mat_mul_tiling_fp32(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    //  1 x 4096, 4096 x 4096
    //  1 x 4096, 4096 x 151936
    int m = A->row, n = B->column, k = A->column; 
    int tile_size = 32;
    // int remain = n % tile_size;
    // if (remain != 0)
    // {
    //     printf("\e[31m[INFO]\e[m tiling size\n");
    //     throw("tiling size error");
    // }

    for (int tile_start = 0; tile_start < n; tile_start += tile_size)
        for (int i = 0; i < m; i++) 
        {
            for (int j = 0; j < tile_size; j++) 
            {
                float acc = 0;
                for (int kk = 0; kk < k; kk++) {
                    acc += data_A[i * k + kk] * data_B[(tile_start + j) * k + kk];
                }
                data_C[i * n + tile_start + j] = acc;
            }
        }
}

void MatmulOperator::matMul_int4_Tiling1vl_qwen(struct qwen_matmul_params *params)
{
    // printf("matMul_int4_Tiling1vl_qwen\n");
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    //
    // step1: quantize activation from fp32 to int8.
    // We have precomputed scaling factors.
    // TODOink: Figure out this function
    int quantization_block_size = params->block_size;
    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->length(), quantization_block_size);
    
    // step2: tile matrix C along the column dimension
    int m = A->row;
    int k = A->column;
    int n = C->column;
    const int tileSize = 64;
    assert (n % tileSize == 0);
    // Each element in the int8 weight matrix contains 2 int4 weights 
    // from adjacent blocks. TODOink: Why do this?
    assert (k % (quantization_block_size * 2) == 0);
    
    // step3: matmul with A8W4
    for (int tile_col_idx = 0; tile_col_idx < n; tile_col_idx += tileSize)
    for (int row_idx = 0; row_idx < m; row_idx++)
    for (int col_idx = 0; col_idx < tileSize; col_idx++)
    {
        // A: (m, k) tile_of_B: (k, tile_size)
        // We need to treat each `blocksize` elements as a group,
        // as they share the same quantization parameters
        float sum = 0;
        for (int ch = 0; ch < k;)
        {

            int8_t * a_int8_ptr = A->int8_data_ptr + row_idx * k + ch;
            // (ch, col_idx + tile_col_idx)
            uint8_t * w_int4_ptr = B->int4_data_ptr + (  (tile_col_idx + col_idx) * k + ch   ) / 2;
            
            float  s_a         = params->A_scales[( row_idx * k + ch ) / quantization_block_size];
            float  s_w         = params->scales[((tile_col_idx + col_idx) * k + ch ) / quantization_block_size];
            int8_t zero_w      = params->zero_point[((tile_col_idx + col_idx) * k + ch ) / quantization_block_size];
            float  s_a_next    = params->A_scales [( row_idx * k + ch ) / quantization_block_size + 1];
            float  s_w_next    = params->scales [((tile_col_idx + col_idx) * k + ch ) / quantization_block_size + 1];
            int8_t zero_w_next = params->zero_point [((tile_col_idx + col_idx) * k + ch ) / quantization_block_size + 1];
            
            int inter_sum = 0;
            int inter_sum_next = 0;
            for (int ch_block = 0; ch_block < quantization_block_size; ch_block++)
            {
                // unpack two int4 weights from one int8
                int8_t w_ =  (w_int4_ptr[ch_block]) & (0x0f);
                int8_t w_next = (w_int4_ptr[ch_block] >> 4);
                int8_t a_ = a_int8_ptr[ch_block];
                int8_t a_next = a_int8_ptr[ch_block + quantization_block_size];
                
                inter_sum += (w_ - zero_w) * (a_);
                inter_sum_next += (w_next - zero_w_next) * (a_next);
            }
            sum += (float) inter_sum * s_w * s_a;
            sum += (float) inter_sum_next * s_w_next * s_a_next;
            ch += quantization_block_size * 2;
        }
        C->data_ptr[row_idx * n + (tile_col_idx + col_idx)] = sum;
    }
    
    
}
}  // namespace matmul
