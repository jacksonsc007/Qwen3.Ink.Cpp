#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "matmul.h"
#include "common.h"

struct multithreading_thread_args {
    int start, end;
    const struct matmul_params* params;
};

struct qwen_multithreading_thread_args {
    int start, end;
    const struct qwen_matmul_params* params;
};


// #define QM_x86
static void* multithreading_worker_func(void* args) {
    struct multithreading_thread_args* mat_args = (struct multithreading_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start; col < mat_args->end; col++) {
            float acc = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int4 weights
                uint8_t* w_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                // pointer of the int8 activation
                const signed char* a_int8 = &A->int8_data_ptr[row * k + ch];
                // scale of weight
                float s_w = params->scales[(col * k + ch) / block_size];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];
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
                // process 16 bytes of weigths (128 bit) = 1 block
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum = 0;
                // process 16 bytes of weigths (128 bit)
                for (int qj = 0; qj < 16; qj++) {
                    // decode a packed byte into two int8 in the range of (-8, 7)
                    uint8_t packed_int4_0 = w_int4[qj];
                    signed char w_de_0 = (packed_int4_0 & 0x0F) - 8.0;
                    signed char w_de_16 = (packed_int4_0 >> 4) - 8.0;
                    // int8 multiply and accumulate operation
                    intermediate_sum += a_int8[qj] * w_de_0;
                    intermediate_sum += a_int8[qj + 16] * w_de_16;
                }
                // dequantize the sum into floating point
                acc += (float)intermediate_sum * s_a * s_w;
                ch += block_size;
#endif
#ifdef QM_x86
                // scales of the second block
                float s_w_2nd = params->scales[(col * k + ch) / block_size + 1];
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
                // process 32 bytes of weigths (256 bit) = 2 blocks
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum = 0, intermediate_sum_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {
                    // decode a packed byte into two int8 in the range of (-8, 7)
                    uint8_t packed_int4_0 = w_int4[qj];
                    signed char w_de_0 = (packed_int4_0 & 0x0F) - 8.0;
                    signed char w_de_16 = (packed_int4_0 >> 4) - 8.0;
                    // int8 multiply and accumulate operation
                    intermediate_sum += a_int8[qj] * w_de_0;
                    intermediate_sum_2nd += a_int8[qj + 32] * w_de_16;
                }
                // dequantize the sum into floating point
                acc += (float)intermediate_sum * s_a * s_w;
                acc += (float)intermediate_sum_2nd * s_a_2nd * s_w_2nd;
                ch += block_size * 2;
#endif
            }
            C->data_ptr[row * n + col] = acc;
        }
    }
    return NULL;
}

 void *multithreading_worker_func_fp32(void *args)
{
    struct multithreading_thread_args *mat_args =
        (struct multithreading_thread_args *) args;
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
            for (int ch = 0; ch < k; ch++)
            {
                sum0 += A->data_ptr[row * k + ch]     * B->data_ptr[(col) * k + ch];
            }
            C->data_ptr[row * n + col] = sum0;
        }
    // NOTE: If we forget to add the following line, the program works in debug mode,
    // but crashes in release mode.
    return NULL;
} 

static void * qwen_int4_multi_thread_func(void * args_)
{
    qwen_multithreading_thread_args * args = (qwen_multithreading_thread_args *)args_;
    int start_col = args -> start;
    int end_col = args -> end;
    const qwen_matmul_params * params = args -> params;
    const matrix * A = &params -> A;
    const matrix * B = &params -> B;
    const matrix * C = &params -> C;
    int m = C->row, n = C->column, k = A->column;
    const int block_size = params -> block_size;
    
    // A: m x k; B: n x k; C: m x n
    // printf("\e[31m[INFO]\e[m reference matrixmul begins\n");
    for (int row = 0; row < m; row++) {
        for (int col = start_col; col < end_col; col++) {
            float acc = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int4 weights
                uint8_t *w_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                // float *w_fp32 = &B->data_ptr[(col * k + ch)];

                // pointer of the int8 activation
                const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];
                // float * a_fp32 = &A->data_ptr[row * k + ch];
                
                
                // scale of weight
                float s_w = params->scales[(col * k + ch) / block_size];
                int8_t zero_w = params->zero_point[(col * k + ch) / block_size];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];
                // scales of the second block
                float s_w_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_a_2nd = params->A_scales[(row * k + ch) / block_size + 1];
                int8_t zero_w_2nd = params->zero_point[(col * k + ch) / block_size + 1];
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
                float intermediate_sum = 0, intermediate_sum_2nd = 0;
                for (int qj = 0; qj < 32; qj++) {

                    uint8_t packed_int4_0 = w_int4[qj];
                    signed char w_de_0 = (int8_t) (packed_int4_0 & 0x0F) - zero_w;
                    signed char w_de_16 = (int8_t) (packed_int4_0 >> 4) - zero_w_2nd;
                    intermediate_sum += a_int8[qj] * w_de_0;
                    intermediate_sum_2nd += a_int8[qj + 32] * w_de_16;
                }
                acc += intermediate_sum * s_w * s_a;
                acc += intermediate_sum_2nd * s_w_2nd * s_a_2nd;
                ch += block_size * 2;
            }
            // if (acc < -20)
            //     printf("\e[31m[ERR]\e[m too big maganitude for row: %d, col: %d, acc: %f\n", row, col, acc);
            C->data_ptr[row * n + col] = acc;
        }
    }
    return NULL;
}



namespace matmul {

void MatmulOperator::matMul_int4_multiThread_qwen(struct qwen_matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32
    float *scale = params->scales, *offset = params->offset;
    // printf("\e[31m[INFO]\e[m mt qwen int4\n");
    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    const int n_threads = 16;
    const int cols_per_thread = (n / n_threads);

    pthread_t threads[n_threads];
    qwen_multithreading_thread_args qwen_th_args[n_threads];
    for (int i = 0; i < n_threads; i++)
    {
        qwen_th_args[i].start = (i)     * cols_per_thread;
        qwen_th_args[i].end   = (i + 1) * cols_per_thread;
        qwen_th_args[i].params = params;
        pthread_create(&threads[i], NULL, qwen_int4_multi_thread_func, &qwen_th_args[i]);
    }
    for (int i = 0; i < n_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    // printf("\e[31m[INFO]\e[m reference matrixmul ends\n");
}


void MatmulOperator::mat_mul_multithreading_fp32(struct matmul_params* params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    int m = C->row, n = C->column, k = A->column;

    const int num_thread = 4;
    pthread_t thread_pool[num_thread];
    struct multithreading_thread_args threads_args[num_thread];
    // printf("\e[31m[INFO]\e[m Multithreading version\n");

    // TODO: Thread creation
    int cols_per_thread =  n / num_thread;
    for (int i = 0; i < num_thread; i++)
    {
        threads_args[i].params = params;
        threads_args[i].start = i * cols_per_thread;
        threads_args[i].end = (i + 1) * cols_per_thread;
        pthread_create(&thread_pool[i], NULL, multithreading_worker_func_fp32, &threads_args[i]);
    }

    // TODO: Join threads
    for (int j = 0; j < num_thread; j++)
    {
        pthread_join(thread_pool[j], NULL);
    }
};

void MatmulOperator::mat_mul_multithreading(struct matmul_params* params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;

    const int num_thread = 4;
    pthread_t thread_pool[num_thread];
    struct multithreading_thread_args threads_args[num_thread];

    // TODO: Thread creation
    int cols_per_thread =  n / num_thread;
    for (int i = 0; i < num_thread; i++)
    {
        threads_args[i].params = params;
        threads_args[i].start = i * cols_per_thread;
        threads_args[i].end = (i + 1) * cols_per_thread;
        pthread_create(&thread_pool[i], NULL, multithreading_worker_func, &threads_args[i]);
    }

    // TODO: Join threads
    for (int j = 0; j < num_thread; j++)
    {
        pthread_join(thread_pool[j], NULL);
    }
};
}  // namespace matmul
