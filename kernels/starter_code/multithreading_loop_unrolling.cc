#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"
struct multithreading_loop_unrolling_thread_args {
    int start, end;
    const struct matmul_params *params;
};
// #define QM_x86
static void *multithreading_loop_unrolling_worker_func(void *args) {
    struct multithreading_loop_unrolling_thread_args *mat_args =
        (struct multithreading_loop_unrolling_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start; col < mat_args->end; col += 4) {
            float acc0 = 0;
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int8 activation
                const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];
                // pointer of the int4 weights
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
                // QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
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
                int8_t mask = 0x0F;
                for (int qj = 0; qj < 32; qj++) {
                    // TODO: decode a packed byte into two int8 in the range of (-8, 7)
                    int8_t w_col0_blk0 = (w0_int4[qj] & mask) - 8;
                    int8_t w_col0_blk1 = (w0_int4[qj] >> 4) - 8;
                    int8_t w_col1_blk0 = (w1_int4[qj] & mask) - 8;
                    int8_t w_col1_blk1 = (w1_int4[qj] >> 4) - 8;
                    int8_t w_col2_blk0 = (w2_int4[qj] & mask) - 8;
                    int8_t w_col2_blk1 = (w2_int4[qj] >> 4) - 8;
                    int8_t w_col3_blk0 = (w3_int4[qj] & mask) - 8;
                    int8_t w_col3_blk1 = (w3_int4[qj] >> 4) - 8;
                    
                    // TODO: int8 multiply and accumulate operation
                    intermediate_sum0 += a_int8[qj] * w_col0_blk0;
                    intermediate_sum1 += a_int8[qj] * w_col1_blk0;
                    intermediate_sum2 += a_int8[qj] * w_col2_blk0;
                    intermediate_sum3 += a_int8[qj] * w_col3_blk0;

                    intermediate_sum0_2nd += a_int8[qj + block_size] * w_col0_blk1;
                    intermediate_sum1_2nd += a_int8[qj + block_size] * w_col1_blk1;
                    intermediate_sum2_2nd += a_int8[qj + block_size] * w_col2_blk1;
                    intermediate_sum3_2nd += a_int8[qj + block_size] * w_col3_blk1;

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
                ch += block_size * 2;
#endif
            }
            C->data_ptr[row * n + col] = acc0;
            C->data_ptr[row * n + col + 1] = acc1;
            C->data_ptr[row * n + col + 2] = acc2;
            C->data_ptr[row * n + col + 3] = acc3;
        }
    }
    return NULL;
}

 void *multithreading_loop_unrolling_worker_func_fp32(void *args)
{
    struct multithreading_loop_unrolling_thread_args *mat_args =
        (struct multithreading_loop_unrolling_thread_args *) args;
    const struct matmul_params *params = mat_args -> params;
    const int start_col_id = mat_args -> start;
    const int end_col_id = mat_args -> end;
    
    // printf("\e[31m[INFO]\e[m Fp32 unrolling version\n");
    // A: activation        (m, k) 
    // B: weight with transposed mem layout (k, n)
    // C: result            (m, n)
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    for (int row = 0; row < m; row++)
        for (int col = start_col_id; col < end_col_id; col++)
        {
            float sum0 = 0.0;
            float sum1 = 0.0;
            float sum2 = 0.0;
            float sum3 = 0.0;
            int ch = 0;
            for (; ch < k - 4;)
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
    // NOTE: If we forget to add the following line, the program works in debug mode,
    // but crashes in release mode.
    return NULL;
} 
namespace matmul {

void MatmulOperator::mat_mul_loop_unrolling4x4_mt_fp32(struct matmul_params *params)
{
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    assert(k == B->row);
    const int num_threads = 8;
    int cols_per_thread = n / num_threads;
    int remaining_cols = n % num_threads;
    if (remaining_cols != 0)
    {
        std::cerr << "columns should be a multple of num_threads" << std::endl;
        throw("columns should be a multple of num_threads");
        // throw std::runtime_error("remaining_cols == 0");
    }

    // NOTE: In release mode, assertion is disabled.
    // assert (n % num_threads == 0);
    pthread_t thread_pool[num_threads];
    struct multithreading_loop_unrolling_thread_args thread_args[num_threads];
    for (int i = 0; i < num_threads; i++) {
        thread_args[i].params = params;
        thread_args[i].start = i * n / num_threads;
        thread_args[i].end = (i + 1) * n / num_threads;
        
        pthread_create(&thread_pool[i], NULL, multithreading_loop_unrolling_worker_func_fp32, &thread_args[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(thread_pool[i], NULL);
    }
}



void MatmulOperator::mat_mul_multithreading_loop_unrolling(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    assert(params->block_size % 32 == 0);  // support block size to be multiples of 32
    assert(A->row == C->row);              // support block size to be multiples of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;

    const int num_thread = 4;
    pthread_t thread_pool[num_thread];
    struct multithreading_loop_unrolling_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    // TODO: Thread creation
    int cols_per_thread = n / num_thread;
    // printf("cols_per_thread: %d\n", cols_per_thread);
    for (int i = 0; i < num_thread; i++)
    {
        // prepare arguments for each threads
        // For A x W, we split W by colums, resulting in [m, k] x [k, n / 4]
        threads_args[i].start =  cols_per_thread * i;
        threads_args[i].end =  cols_per_thread * (i + 1);
        threads_args[i].params = params;
        
        pthread_create(&thread_pool[i], NULL, multithreading_loop_unrolling_worker_func, &threads_args[i]);
    }

    // TODO: Join threads
    for (int j = 0; j < num_thread; j++)
    {
        pthread_join(thread_pool[j], NULL);
    }
};
}  // namespace matmul
