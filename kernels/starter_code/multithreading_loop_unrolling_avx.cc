#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <cmath>
#include <cstdlib>
// include avx intrinsics
#include <immintrin.h>
#include <xmmintrin.h>

#include "../matmul.h"
#include "common.h"

struct multithreading_loop_unrolling_thread_args {
    int start, end;
    const struct matmul_params *params;
};
// #define QM_x86

 void *multithreading_loop_unrolling_worker_avx_func_fp32(void *args)
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
    __m256 zero256 = _mm256_setzero_ps();

    for (int row = 0; row < m; row++)
        for (int col = start_col_id; col < end_col_id; col++)
        {
            __m256 acc0 = zero256;
            __m256 acc1 = zero256;
            __m256 acc2 = zero256;
            __m256 acc3 = zero256;
            
            int ch = 0;
            for (; ch <= k - 4 * 8;)
            {
                // __m256 *A256_0 = (__m256 *) &A->data_ptr[row * k + ch];
                // __m256 *B256_0 = (__m256 *) &B->data_ptr[col * k + ch];
                // __m256 *A256_1 = (__m256 *) &A->data_ptr[row * k + ch + 8];
                // __m256 *B256_1 = (__m256 *) &B->data_ptr[col * k + ch + 8];
                // __m256 *A256_2 = (__m256 *) &A->data_ptr[row * k + ch + 16];
                // __m256 *B256_2 = (__m256 *) &B->data_ptr[col * k + ch + 16];
                // __m256 *A256_3 = (__m256 *) &A->data_ptr[row * k + ch + 24];
                // __m256 *B256_3 = (__m256 *) &B->data_ptr[col * k + ch + 24];

                // get address
                __m256 *A256_0 = (__m256 *) (A->data_ptr + row * k + ch      );
                __m256 *B256_0 = (__m256 *) (B->data_ptr + col * k + ch      );
                __m256 *A256_1 = (__m256 *) (A->data_ptr + row * k + ch + 8  );
                __m256 *B256_1 = (__m256 *) (B->data_ptr + col * k + ch + 8  );
                __m256 *A256_2 = (__m256 *) (A->data_ptr + row * k + ch + 16 );
                __m256 *B256_2 = (__m256 *) (B->data_ptr + col * k + ch + 16 );
                __m256 *A256_3 = (__m256 *) (A->data_ptr + row * k + ch + 24 );
                __m256 *B256_3 = (__m256 *) (B->data_ptr + col * k + ch + 24 );
                
                // load packed values
                __m256 packed_a_0 = _mm256_load_ps((const float *) A256_0);
                __m256 packed_b_0 = _mm256_load_ps((const float *) B256_0);
                __m256 packed_a_1 = _mm256_load_ps((const float *) A256_1);
                __m256 packed_b_1 = _mm256_load_ps((const float *) B256_1);
                __m256 packed_a_2 = _mm256_load_ps((const float *) A256_2);
                __m256 packed_b_2 = _mm256_load_ps((const float *) B256_2);
                __m256 packed_a_3 = _mm256_load_ps((const float *) A256_3);
                __m256 packed_b_3 = _mm256_load_ps((const float *) B256_3);
                
                // mul and add
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(packed_a_0, packed_b_0));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(packed_a_1, packed_b_1));
                acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(packed_a_2, packed_b_2));
                acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(packed_a_3, packed_b_3));

                ch += 4 * 8;
            }

            float rest_sum = 0;
            for (; ch < k; ch++)
            {
                rest_sum += A->data_ptr[row * k + ch] * B->data_ptr[col * k + ch];
            }
            
            // summing up
            float * ptr0 = (float *)&acc0; 
            float * ptr1 = (float *)&acc1;
            float * ptr2 = (float *)&acc2;
            float * ptr3 = (float *)&acc3;
            float all_sum = rest_sum;

            all_sum  += ptr0[0] + ptr0[1] + ptr0[2] + ptr0[3] + 
                        ptr0[4] + ptr0[5] + ptr0[6] + ptr0[7];
            all_sum  += ptr1[0] + ptr1[1] + ptr1[2] + ptr1[3] + 
                        ptr1[4] + ptr1[5] + ptr1[6] + ptr1[7];
            all_sum  += ptr2[0] + ptr2[1] + ptr2[2] + ptr2[3] + 
                        ptr2[4] + ptr2[5] + ptr2[6] + ptr2[7];
            all_sum  += ptr3[0] + ptr3[1] + ptr3[2] + ptr3[3] + 
                        ptr3[4] + ptr3[5] + ptr3[6] + ptr3[7];
            C->data_ptr[row * n + col] = all_sum;
        }

    // NOTE: If we forget to add the following line, the program works in debug mode,
    // but crashes in release mode.
    return NULL;
} 
namespace matmul {

void MatmulOperator::mat_mul_loop_unrolling4x4_mt_avx_fp32(struct matmul_params *params)
{
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    assert(k == B->row);
    const int num_threads = 4;
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
        
        pthread_create(&thread_pool[i], NULL, multithreading_loop_unrolling_worker_avx_func_fp32, &thread_args[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(thread_pool[i], NULL);
    }
}



}  // namespace matmul
