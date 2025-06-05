#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "matmul.h"
#include "common.h"


namespace matmul {


void MatmulOperator::matMul_int4_unrolling2x2_qwen(struct qwen_matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32
    float *scale = params->scales, *offset = params->offset;
    // printf("\e[31m[INFO]\e[m int4 unroll2x2 begins\n");

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    // printf("\e[31m[INFO]\e[m reference matrixmul begins\n");
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col+=2) {
            float acc0 = 0;
            float acc1 = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int4 weights
                uint8_t *w_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                uint8_t *w_int4_1 = &B->int4_data_ptr[((col+1) * k + ch) / 2];
                // float *w_fp32 = &B->data_ptr[(col * k + ch)];

                // pointer of the int8 activation
                const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];
                // float * a_fp32 = &A->data_ptr[row * k + ch];
                
                
                // scale of weight
                float s_w = params->scales[(col * k + ch) / block_size];
                float s_w_1 = params->scales[((col+1) * k + ch) / block_size];
                int8_t zero_w = params->zero_point[(col * k + ch) / block_size];
                int8_t zero_w_1 = params->zero_point[((col+1) * k + ch) / block_size];

                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];

                // scales of the second block
                float s_w_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_w_2nd_1 = params->scales[((col+1) * k + ch) / block_size + 1];
                float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];
                int8_t zero_w_2nd = params->zero_point[(col * k + ch) / block_size + 1];
                int8_t zero_w_2nd_1 = params->zero_point[((col+1) * k + ch) / block_size + 1];
                // order of weights with QM_x86:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
                // QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         256 bit
                // process 32 bytes of weigths (256 bit) = 2 blocks
                // intermediate variable to store sum of integer multiplication and accumulation
                float intermediate_sum_0_0 = 0, intermediate_sum_0_1 = 0;
                float intermediate_sum_1_0 = 0, intermediate_sum_1_1 = 0;
                for (int qj = 0; qj < 32; qj++) {

                    // 1st roll
                    uint8_t packed_int4_0 = w_int4[qj];
                    signed char w_de_0 = (int8_t) (packed_int4_0 & 0x0F) - zero_w;
                    signed char w_de_16 = (int8_t) (packed_int4_0 >> 4) - zero_w_2nd;
                    intermediate_sum_0_0 += a_int8[qj] * w_de_0;
                    intermediate_sum_0_1 += a_int8[qj + 32] * w_de_16;
                    
                    // 2nd roll
                    uint8_t packed_int4_1 = w_int4_1[qj];
                    signed char w_de_0_1 = (int8_t) (packed_int4_1 & 0x0F) - zero_w_1;
                    signed char w_de_16_1 = (int8_t) (packed_int4_1 >> 4) - zero_w_2nd_1;
                    intermediate_sum_1_0 += a_int8[qj] * w_de_0_1;
                    intermediate_sum_1_1 += a_int8[qj + 32] * w_de_16_1;

                }
                acc0 += intermediate_sum_0_0 * s_w * s_a;
                acc0 += intermediate_sum_0_1 * s_w_2nd * s_a_2nd;

                acc1 += intermediate_sum_1_0 * s_w_1 * s_a;
                acc1 += intermediate_sum_1_1 * s_w_2nd_1 * s_a_2nd;

                ch += block_size * 2;

            }
            C->data_ptr[row * n + col] = acc0;
            C->data_ptr[row * n + col + 1] = acc1;
        }
    }
    // printf("\e[31m[INFO]\e[m reference matrixmul ends\n");
}


void MatmulOperator::mat_mul_loop_unrolling8x8_fp32(struct matmul_params *params) 
{
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    
    // Loop over rows of matrix C
    for (int row = 0; row < m; row++)
        // Loop over columns of matrix C
        for (int col = 0; col < n; col++)
        {
            // Accumulate sums for 8 terms
            float sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            float sum4 = 0.0, sum5 = 0.0, sum6 = 0.0, sum7 = 0.0;

            int ch = 0;
            // Loop over the columns of A (and rows of B), unrolled by 8
            int row_x_k = row * k;
            int col_x_k = col * n;
            for (; ch < k - 8; ch += 8)
            {
                // Process 8 values at a time
                sum0 += A->data_ptr[row_x_k + ch]     * B->data_ptr[col_x_k+ ch];
                sum1 += A->data_ptr[row_x_k + ch + 1] * B->data_ptr[col_x_k+ ch + 1];
                sum2 += A->data_ptr[row_x_k + ch + 2] * B->data_ptr[col_x_k+ ch + 2];
                sum3 += A->data_ptr[row_x_k + ch + 3] * B->data_ptr[col_x_k+ ch + 3];
                sum4 += A->data_ptr[row_x_k + ch + 4] * B->data_ptr[col_x_k+ ch + 4];
                sum5 += A->data_ptr[row_x_k + ch + 5] * B->data_ptr[col_x_k+ ch + 5];
                sum6 += A->data_ptr[row_x_k + ch + 6] * B->data_ptr[col_x_k+ ch + 6];
                sum7 += A->data_ptr[row_x_k + ch + 7] * B->data_ptr[col_x_k+ ch + 7];
            }
            // Handle remaining elements if k is not a multiple of 8
            for (; ch < k; ch++)
            {
                sum0 += A->data_ptr[row * k + ch] * B->data_ptr[col * n + ch];
            }
            
            // Store the final result in C
            C->data_ptr[row * n + col] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
        }
}
void MatmulOperator::mat_mul_loop_unrolling4x1_fp32(struct matmul_params *params) 
{
    
    // printf("\e[31m[INFO]\e[m Fp32 unrolling version\n");
    // A: activation        (m, k) 
    // B: transposed weight (k, n)
    // C: result            (m, n)
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    // printf("[INFO] matmul_loop_unrolling4x4_fp32: m=%d, n=%d, k=%d\n", m, n, k);
    for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
        {
            float sum0 = 0.0;
            int ch = 0;
            for (; ch < k - 4;)
            {
                sum0 += A->data_ptr[row * k + ch]     * B->data_ptr[(col) * k + ch   ];
                sum0 += A->data_ptr[row * k + ch + 1] * B->data_ptr[(col) * k + ch + 1];
                sum0 += A->data_ptr[row * k + ch + 2] * B->data_ptr[(col) * k + ch + 2];
                sum0 += A->data_ptr[row * k + ch + 3] * B->data_ptr[(col) * k + ch + 3];
                ch += 4;
            }
            for (; ch < k; ch++)
            {
                // printf("\e[31m[INFO]\e[m dealing with remaining channels...\n");
                sum0 += A->data_ptr[row * k + ch]     * B->data_ptr[(col) * k + ch];

            }
            C->data_ptr[row * n + col] = sum0;
        }
}

void MatmulOperator::matMul_loopUnrolling2level4x4_fp32(struct matmul_params *params) 
{
    
    // printf("\e[31m[INFO]\e[m Fp32 unrolling version\n");
    // A: activation        (m, k) 
    // B: transposed weight (k, n)
    // C: result            (m, n)
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    // printf("[INFO] matmul_loop_unrolling 2level innermost: m=%d, n=%d, k=%d\n", m, n, k);
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col+=4)
        {
            float sum0 = 0.0;
            float sum1 = 0.0;
            float sum2 = 0.0;
            float sum3 = 0.0;

            for (int ch = 0; ch < k ; ch+=4)
            {
                // sum0 += A->data_ptr[row * k + ch    ]     * B->data_ptr[(col    ) * k + ch   ];
                // sum1 += A->data_ptr[row * k + ch    ]     * B->data_ptr[(col + 1) * k + ch   ];
                // sum2 += A->data_ptr[row * k + ch    ]     * B->data_ptr[(col + 2) * k + ch   ];
                // sum3 += A->data_ptr[row * k + ch    ]     * B->data_ptr[(col + 3) * k + ch   ];
                
                // sum0 += A->data_ptr[row * k + ch + 1]     * B->data_ptr[(col    ) * k + ch + 1];
                // sum1 += A->data_ptr[row * k + ch + 1]     * B->data_ptr[(col + 1) * k + ch + 1];
                // sum2 += A->data_ptr[row * k + ch + 1]     * B->data_ptr[(col + 2) * k + ch + 1];
                // sum3 += A->data_ptr[row * k + ch + 1]     * B->data_ptr[(col + 3) * k + ch + 1];
                
                // sum0 += A->data_ptr[row * k + ch + 2]     * B->data_ptr[(col    ) * k + ch + 2];
                // sum1 += A->data_ptr[row * k + ch + 2]     * B->data_ptr[(col + 1) * k + ch + 2];
                // sum2 += A->data_ptr[row * k + ch + 2]     * B->data_ptr[(col + 2) * k + ch + 2];
                // sum3 += A->data_ptr[row * k + ch + 2]     * B->data_ptr[(col + 3) * k + ch + 2];
                
                // sum0 += A->data_ptr[row * k + ch + 3]     * B->data_ptr[(col    ) * k + ch + 3];
                // sum1 += A->data_ptr[row * k + ch + 3]     * B->data_ptr[(col + 1) * k + ch + 3];
                // sum2 += A->data_ptr[row * k + ch + 3]     * B->data_ptr[(col + 2) * k + ch + 3];
                // sum3 += A->data_ptr[row * k + ch + 3]     * B->data_ptr[(col + 3) * k + ch + 3];

                //////////////////////////////////////////////////////////////////////////////////
                // sum0 += A->data_ptr[row * k + ch    ]     * B->data_ptr[(col    ) * k + ch   ];
                // sum0 += A->data_ptr[row * k + ch + 1]     * B->data_ptr[(col    ) * k + ch + 1];
                // sum0 += A->data_ptr[row * k + ch + 2]     * B->data_ptr[(col    ) * k + ch + 2];
                // sum0 += A->data_ptr[row * k + ch + 3]     * B->data_ptr[(col    ) * k + ch + 3];
                
                // sum1 += A->data_ptr[row * k + ch    ]     * B->data_ptr[(col + 1) * k + ch   ];
                // sum1 += A->data_ptr[row * k + ch + 1]     * B->data_ptr[(col + 1) * k + ch + 1];
                // sum1 += A->data_ptr[row * k + ch + 2]     * B->data_ptr[(col + 1) * k + ch + 2];
                // sum1 += A->data_ptr[row * k + ch + 3]     * B->data_ptr[(col + 1) * k + ch + 3];

                // sum2 += A->data_ptr[row * k + ch    ]     * B->data_ptr[(col + 2) * k + ch   ];
                // sum2 += A->data_ptr[row * k + ch + 1]     * B->data_ptr[(col + 2) * k + ch + 1];
                // sum2 += A->data_ptr[row * k + ch + 2]     * B->data_ptr[(col + 2) * k + ch + 2];
                // sum2 += A->data_ptr[row * k + ch + 3]     * B->data_ptr[(col + 2) * k + ch + 3];

                // sum3 += A->data_ptr[row * k + ch    ]     * B->data_ptr[(col + 3) * k + ch   ];
                // sum3 += A->data_ptr[row * k + ch + 1]     * B->data_ptr[(col + 3) * k + ch + 1];
                // sum3 += A->data_ptr[row * k + ch + 2]     * B->data_ptr[(col + 3) * k + ch + 2];
                // sum3 += A->data_ptr[row * k + ch + 3]     * B->data_ptr[(col + 3) * k + ch + 3];
            }

            C->data_ptr[row * n + col    ] = sum0;
            C->data_ptr[row * n + col + 1] = sum1;
            C->data_ptr[row * n + col + 2] = sum2;
            C->data_ptr[row * n + col + 3] = sum3;
        }
    }
}
void MatmulOperator::mat_mul_loop_unrolling_second_innermost_4x4_fp32(struct matmul_params *params) 
{
    
    // printf("\e[31m[INFO]\e[m Fp32 unrolling version\n");
    // A: activation        (m, k) 
    // B: transposed weight (k, n)
    // C: result            (m, n)
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    // printf("[INFO] matmul_loop_unrolling second innermost: m=%d, n=%d, k=%d\n", m, n, k);
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col+=4)
        {
            float sum0 = 0.0;
            float sum1 = 0.0;
            float sum2 = 0.0;
            float sum3 = 0.0;

            for (int ch = 0; ch < k ; ch++)
            {
                sum0 += A->data_ptr[row * k + ch]     * B->data_ptr[(col    ) * k + ch   ];
                sum1 += A->data_ptr[row * k + ch]     * B->data_ptr[(col + 1) * k + ch   ];
                sum2 += A->data_ptr[row * k + ch]     * B->data_ptr[(col + 2) * k + ch   ];
                sum3 += A->data_ptr[row * k + ch]     * B->data_ptr[(col + 3) * k + ch   ];
            }

            C->data_ptr[row * n + col    ] = sum0;
            C->data_ptr[row * n + col + 1] = sum1;
            C->data_ptr[row * n + col + 2] = sum2;
            C->data_ptr[row * n + col + 3] = sum3;
        }
    }
}
void MatmulOperator::mat_mul_loop_unrolling4x4_fp32(struct matmul_params *params) 
{
    
    // printf("\e[31m[INFO]\e[m Fp32 unrolling version\n");
    // A: activation        (m, k) 
    // B: transposed weight (k, n)
    // C: result            (m, n)
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    // printf("[INFO] matmul_loop_unrolling4x4_fp32: m=%d, n=%d, k=%d\n", m, n, k);
    for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
        {
            float sum0 = 0.0;
            float sum1 = 0.0;
            float sum2 = 0.0;
            float sum3 = 0.0;
            int ch = 0;
            for (; ch <= k - 4;)
            {
                sum0 += A->data_ptr[row * k + ch]     * B->data_ptr[(col) * k + ch   ];
                sum1 += A->data_ptr[row * k + ch + 1] * B->data_ptr[(col) * k + ch + 1];
                sum2 += A->data_ptr[row * k + ch + 2] * B->data_ptr[(col) * k + ch + 2];
                sum3 += A->data_ptr[row * k + ch + 3] * B->data_ptr[(col) * k + ch + 3];
                ch += 4;
            }
            for (; ch < k; ch++)
            {
                // printf("\e[31m[INFO]\e[m dealing with remaining channels...\n");
                sum0 += A->data_ptr[row * k + ch]     * B->data_ptr[(col) * k + ch];

            }
            C->data_ptr[row * n + col] = sum0 + sum1 + sum2 + sum3;
        }
}

void MatmulOperator::mat_mul_loop_unrolling(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32
    float *scale = params->scales, *offset = params->offset;

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col += 4) {
            float acc0 = 0;
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int8 activation
                const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];
                // pointer of the int4 weights
                // coordinates in linear layout memory: (ch, col) for matrix of size [k, n]
                // coordinates in int4 transformed memory: (col, ch) for matrix of size [n, k]
                // TODO: why divided by 2?
                uint8_t *w0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                uint8_t *w1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];
                uint8_t *w2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];
                uint8_t *w3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];
                // scale of weight
                float s_w0 = params->scales[(col * k + ch) / block_size];
                float s_w1 = params->scales[((col + 1) * k + ch) / block_size];
                float s_w2 = params->scales[((col + 2) * k + ch) / block_size];
                float s_w3 = params->scales[((col + 3) * k + ch) / block_size];
#ifdef QM_ARM
                // order of weights with QM_ARM:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)
                // QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         128 bit                         127
                // process 16 bytes of weigths (128 bit) = 1 block for each of unrolled `col`
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                for (int qj = 0; qj < 16; qj++) {
                    // TODO: decode a packed byte into two int8 in the range of (-8, 7)

                    // TODO: int8 multiply and accumulate operation
                }
                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;
                ch += block_size;
#endif
#ifdef QM_x86
                // scales of the second block
                float s_w0_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_w1_2nd = params->scales[((col + 1) * k + ch) / block_size + 1];
                float s_w2_2nd = params->scales[((col + 2) * k + ch) / block_size + 1];
                float s_w3_2nd = params->scales[((col + 3) * k + ch) / block_size + 1];
                float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];
                // order of weights with QM_x86:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
                // QM_x86 order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         256 bit
                // process 32 bytes of weigths (256 bit) = 2 blocks for each of unrolled `col`
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                int intermediate_sum0_2nd = 0, intermediate_sum1_2nd = 0, intermediate_sum2_2nd = 0,
                    intermediate_sum3_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {
                    // TODO: decode a packed byte into two int8 in the range of (-8, 7)
                    // NOTE: How to decode a packed byte into two int8
                    // NOTE: is determined by how they are encoded. Please check `quantize_methods.py`
                    uint8_t mask = 0xF;
                    int8_t weight_lower_4bits = (w0_int4[qj] & mask) - 8;
                    int8_t weight_higher_4bits = (w0_int4[qj]>>4) - 8;
                    // make sure the emulated range of int4, i.e. (-8, 7)
                    // int8_t weight_lower_neg = weight_lower_4bits | 0xf0;
                    // if (weight_lower_4bits & 0x08) weight_lower_4bits = weight_lower_neg;
                    // int8_t weight_higher_neg = weight_higher_4bits | 0xf0;
                    // if (weight_higher_4bits & 0x08) weight_higher_4bits = weight_higher_neg;
                    // TODO: int8 multiply and accumulate operation
                    intermediate_sum0 += a_int8[qj] * weight_lower_4bits;
                    intermediate_sum0_2nd += a_int8[qj + block_size] * weight_higher_4bits;
                    
                    // col-1
                    weight_lower_4bits = (w1_int4[qj] & mask) - 8;
                    weight_higher_4bits = (w1_int4[qj]>>4) - 8;
                    // make sure the emulated range of int4, i.e. (-8, 7)
                    // weight_lower_neg = weight_lower_4bits | 0xf0;
                    // if (weight_lower_4bits & 0x08) weight_lower_4bits = weight_lower_neg;
                    // weight_higher_neg = weight_higher_4bits | 0xf0;
                    // if (weight_higher_4bits & 0x08) weight_higher_4bits = weight_higher_neg;
                    // TODO: int8 multiply and accumulate operation
                    intermediate_sum1 += a_int8[qj] * weight_lower_4bits;
                    intermediate_sum1_2nd += a_int8[qj + block_size] * weight_higher_4bits;

                    // col-2
                    weight_lower_4bits = (w2_int4[qj] & mask) - 8;
                    weight_higher_4bits = (w2_int4[qj]>>4) - 8;
                    // make sure the emulated range of int4, i.e. (-8, 7)
                    // weight_lower_neg = weight_lower_4bits | 0xf0;
                    // if (weight_lower_4bits & 0x08) weight_lower_4bits = weight_lower_neg;
                    // weight_higher_neg = weight_higher_4bits | 0xf0;
                    // if (weight_higher_4bits & 0x08) weight_higher_4bits = weight_higher_neg;
                    // TODO: int8 multiply and accumulate operation
                    intermediate_sum2 += a_int8[qj] * weight_lower_4bits;
                    intermediate_sum2_2nd += a_int8[qj + block_size] * weight_higher_4bits;

                    // col-3
                    weight_lower_4bits = (w3_int4[qj] & mask) - 8;
                    weight_higher_4bits = (w3_int4[qj]>>4) - 8;
                    // make sure the emulated range of int4, i.e. (-8, 7)
                    // weight_lower_neg = weight_lower_4bits | 0xf0;
                    // if (weight_lower_4bits & 0x08) weight_lower_4bits = weight_lower_neg;
                    // weight_higher_neg = weight_higher_4bits | 0xf0;
                    // if (weight_higher_4bits & 0x08) weight_higher_4bits = weight_higher_neg;
                    // TODO: int8 multiply and accumulate operation
                    intermediate_sum3 += a_int8[qj] * weight_lower_4bits;
                    intermediate_sum3_2nd += a_int8[qj + block_size] * weight_higher_4bits;

                }
                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc0 += (float)intermediate_sum0_2nd * s_a_2nd * s_w0_2nd;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc1 += (float)intermediate_sum1_2nd * s_a_2nd * s_w1_2nd;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc2 += (float)intermediate_sum2_2nd * s_a_2nd * s_w2_2nd;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;
                acc3 += (float)intermediate_sum3_2nd * s_a_2nd * s_w3_2nd;
                // process two blocks
                ch += block_size * 2;
#endif
            }
            C->data_ptr[row * n + col] = acc0;
            C->data_ptr[row * n + col + 1] = acc1;
            C->data_ptr[row * n + col + 2] = acc2;
            C->data_ptr[row * n + col + 3] = acc3;
        }
    }
};
}  // namespace matmul
