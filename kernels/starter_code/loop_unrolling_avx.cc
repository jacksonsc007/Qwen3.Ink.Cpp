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


namespace matmul {

void MatmulOperator::matMul_loopUnrollingSecondInnermost4x4_avx_fp32(struct matmul_params *params) 
{
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;
    // printf("matMul_loopUnrollingSecondInnermost4x4_avx_fp32\n");

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j+=4) 
        {
            __m256 acc_packed_0 = _mm256_setzero_ps();
            __m256 acc_packed_1 = _mm256_setzero_ps();
            __m256 acc_packed_2 = _mm256_setzero_ps();
            __m256 acc_packed_3 = _mm256_setzero_ps();
            int kk = 0;
            for (; kk <= k-8; kk+=8) 
            {
                // __m256 * a_ptr   = (__m256 *) (data_A + (i    ) * k + kk);
                // __m256 * b_ptr_0 = (__m256 *) (data_B + (j    ) * k + kk);
                // __m256 * b_ptr_1 = (__m256 *) (data_B + (j + 1) * k + kk);
                // __m256 * b_ptr_2 = (__m256 *) (data_B + (j + 2) * k + kk);
                // __m256 * b_ptr_3 = (__m256 *) (data_B + (j + 3) * k + kk);
                
                
                __m256 a_val   = _mm256_loadu_ps(data_A + (i    ) * k + kk);
                __m256 b_val_0 = _mm256_loadu_ps(data_B + (j    ) * k + kk);
                __m256 b_val_1 = _mm256_loadu_ps(data_B + (j + 1) * k + kk);
                __m256 b_val_2 = _mm256_loadu_ps(data_B + (j + 2) * k + kk);
                __m256 b_val_3 = _mm256_loadu_ps(data_B + (j + 3) * k + kk);


                acc_packed_0 = _mm256_add_ps(acc_packed_0, _mm256_mul_ps(a_val, b_val_0));
                acc_packed_1 = _mm256_add_ps(acc_packed_1, _mm256_mul_ps(a_val, b_val_1));
                acc_packed_2 = _mm256_add_ps(acc_packed_2, _mm256_mul_ps(a_val, b_val_2));
                acc_packed_3 = _mm256_add_ps(acc_packed_3, _mm256_mul_ps(a_val, b_val_3));
            }
            
            float * ptr0 = (float*) &acc_packed_0;
            data_C[i * n + j    ] = ptr0[0] + ptr0[1] + ptr0[2] + ptr0[3] + 
                        ptr0[4] + ptr0[5] + ptr0[6] + ptr0[7];

            float * ptr1 = (float*) &acc_packed_1;
            data_C[i * n + j + 1] = ptr1[0] + ptr1[1] + ptr1[2] + ptr1[3] + 
                        ptr1[4] + ptr1[5] + ptr1[6] + ptr1[7];

            float * ptr2 = (float*) &acc_packed_2;
            data_C[i * n + j + 2] = ptr2[0] + ptr2[1] + ptr2[2] + ptr2[3] + 
                        ptr2[4] + ptr2[5] + ptr2[6] + ptr2[7];

            float * ptr3 = (float*) &acc_packed_3;
            data_C[i * n + j + 3] = ptr3[0] + ptr3[1] + ptr3[2] + ptr3[3] + 
                        ptr3[4] + ptr3[5] + ptr3[6] + ptr3[7];
        }
    }
}
void MatmulOperator::mat_mul_loop_unrolling4x4_avx_fp32(struct matmul_params *params) 
{
    
    // printf("\e[31m[INFO]\e[m Fp32 unrolling version\n");
    // A: activation        (m, k) 
    // B: transposed weight (k, n)
    // C: result            (m, n)
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int m = C->row, n = C->column, k = A->column;
    // printf("[INFO] matmul_loop_unrolling4x4_fp32: m=%d, n=%d, k=%d\n", m, n, k);
    __m256 zero256 = _mm256_setzero_ps();
    for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
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
}

}
