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


 void *multithreading_tiling_worker_func_fp32(void *args)
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
    int tile_size = 32;
    int remain = (end_col_id - start_col_id) % tile_size;
    if (remain != 0)
    {
        printf("start: %d, end: %d, tile_size: %d, remain: %d\n", start_col_id, end_col_id, tile_size, remain);
        throw("tiling size error");
    }

    for (int tile_start_idx = start_col_id; tile_start_idx < end_col_id; tile_start_idx += tile_size)
        for (int row = 0; row < m; row++)
            for (int col = 0; col < tile_size; col += 1)
            {
                float sum0 = 0.0;
                for (int ch = 0; ch < k; ch++)
                {
                    sum0 += A->data_ptr[row * k + ch]     * B->data_ptr[(col + tile_start_idx) * k + ch];
                }
                C->data_ptr[row * n + col + tile_start_idx] = sum0;
            }
    // NOTE: If we forget to add the following line, the program works in debug mode,
    // but crashes in release mode.
    return NULL;
} 



namespace matmul {
void MatmulOperator::mat_mul_multithreading_tiling_fp32(struct matmul_params* params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    int m = C->row, n = C->column, k = A->column;

    const int num_thread = 4;
    pthread_t thread_pool[num_thread];
    struct multithreading_thread_args threads_args[num_thread];
    // printf("\e[31m[INFO]\e[m Multithreading tiling version\n");

    // TODO: Thread creation
    int cols_per_thread =  n / num_thread;
    for (int i = 0; i < num_thread; i++)
    {
        threads_args[i].params = params;
        threads_args[i].start = i * cols_per_thread;
        threads_args[i].end = (i + 1) * cols_per_thread;
        pthread_create(&thread_pool[i], NULL, multithreading_tiling_worker_func_fp32, &threads_args[i]);
    }

    // TODO: Join threads
    for (int j = 0; j < num_thread; j++)
    {
        pthread_join(thread_pool[j], NULL);
    }
};

}  // namespace matmul
