#include <cmath>

#include "common.h"
#include "operators.h"

/* void softmax(const Matrix3D<float> &input, Matrix3D<float> &output) {
    PROFILE_START("softmax");
    int len = input.length();
    int n_heads = input.m_dim_x;
    int sqlen = input.m_dim_y;
    int context_len = input.m_dim_z;
    // Find the maximum value in the input array
    for (int i = 0; i < n_heads; i++) {
        for (int j = 0; j < sqlen; j++) {
            float max_value = std::numeric_limits<float>::lowest();
            float sum = 0;
            // Find the maximum value in the input array
            for (int k = 0; k < context_len; k++) {
                float value = input(i, j, k);
                if (value > max_value) {
                    max_value = value;
                }
            }

            // Compute the softmax values
            for (int k = 0; k < context_len; k++) {
                float value = input(i, j, k);
                sum += std::exp(value - max_value);
                // IF_DEBUG_ATTENTION(
                //     printf("sum = %f value = %f, max_value = %f\n", sum, value, max_value);
                // );
            }

            // Normalize the softmax values and store them in the output array
            for (int k = 0; k < context_len; k++) {
                float value = input(i, j, k);
                assert(sum != 0);
                output(i, j, k) = (std::exp(value - max_value) / sum);
            }
        }
    }
    PROFILE_END("softmax");
} */

void softmax(const Matrix3D<float> &input, Matrix3D<float> &output) {
    PROFILE_START("softmax");
    const int n_heads = input.m_dim_x;
    const int sqlen = input.m_dim_y;
    const int context_len = input.m_dim_z;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_heads; i++) {
        for (int j = 0; j < sqlen; j++) {
            // Find max value (first pass)
            float max_val = input(i, j, 0);
            for (int k = 1; k < context_len; k++) {
                max_val = std::max(max_val, input(i, j, k));
            }

            // Compute exp(x - max) and sum (second pass)
            float sum = 0.0f;
            float exps[context_len]; // Stack-allocated for small context_len
            for (int k = 0; k < context_len; k++) {
                exps[k] = std::exp(input(i, j, k) - max_val);
                sum += exps[k];
            }

            // Normalize (third pass)
            const float inv_sum = 1.0f / sum; // Avoid division in loop
            for (int k = 0; k < context_len; k++) {
                output(i, j, k) = exps[k] * inv_sum;
            }
        }
    }
    PROFILE_END("softmax");
}

// #include <immintrin.h>
// #include <cmath>
// #include <cfloat>
// #include <cstdlib>
// #include <stdexcept>

// void softmax(const Matrix3D<float>& input, Matrix3D<float>& output) {
//     PROFILE_START("softmax_avx2");
//     const int n_heads = input.m_dim_x;
//     const int sqlen = input.m_dim_y;
//     const int context_len = input.m_dim_z;

//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < n_heads; i++) {
//         for (int j = 0; j < sqlen; j++) {
//             // Find max value (AVX2 accelerated)
//             __m256 max_vec = _mm256_set1_ps(-FLT_MAX);
//             int k = 0;
//             for (; k <= context_len - 8; k += 8) {
//                 __m256 x = _mm256_loadu_ps(&input(i, j, k));
//                 max_vec = _mm256_max_ps(max_vec, x);
//             }
            
//             // Horizontal max
//             float max_val = -FLT_MAX;
//             float max_arr[8] __attribute__((aligned(32)));
//             _mm256_store_ps(max_arr, max_vec);
//             for (int m = 0; m < 8; m++) {
//                 max_val = std::max(max_val, max_arr[m]);
//             }
            
//             // Handle remainder
//             for (; k < context_len; k++) {
//                 max_val = std::max(max_val, input(i, j, k));
//             }

//             // Compute exp(x - max) and sum
//             float sum = 0.0f;
//             float* exps = static_cast<float*>(aligned_alloc(32, context_len * sizeof(float)));
//             if (!exps) throw std::bad_alloc();
            
//             try {
//                 // Process in chunks of 8 for AVX2
//                 k = 0;
//                 for (; k <= context_len - 8; k += 8) {
//                     __m256 x = _mm256_loadu_ps(&input(i, j, k));
//                     __m256 shifted = _mm256_sub_ps(x, _mm256_set1_ps(max_val));
                    
//                     // Compute exp using scalar operations within SIMD
//                     float shifted_scalar[8] __attribute__((aligned(32)));
//                     _mm256_store_ps(shifted_scalar, shifted);
                    
//                     float exp_scalar[8];
//                     for (int m = 0; m < 8; m++) {
//                         exp_scalar[m] = std::exp(shifted_scalar[m]);
//                         sum += exp_scalar[m];
//                     }
//                     _mm256_store_ps(&exps[k], _mm256_load_ps(exp_scalar));
//                 }
                
//                 // Handle remainder
//                 for (; k < context_len; k++) {
//                     exps[k] = std::exp(input(i, j, k) - max_val);
//                     sum += exps[k];
//                 }

//                 // Handle underflow case
//                 if (sum == 0.0f) {
//                     const float uniform_prob = 1.0f / context_len;
//                     for (k = 0; k < context_len; k++) {
//                         output(i, j, k) = uniform_prob;
//                     }
//                     free(exps);
//                     continue;
//                 }

//                 // Normalize
//                 const float inv_sum = 1.0f / sum;
//                 k = 0;
//                 for (; k <= context_len - 8; k += 8) {
//                     __m256 exp_x = _mm256_load_ps(&exps[k]);
//                     __m256 result = _mm256_mul_ps(exp_x, _mm256_set1_ps(inv_sum));
//                     _mm256_storeu_ps(&output(i, j, k), result);
//                 }
                
//                 // Handle remainder
//                 for (; k < context_len; k++) {
//                     output(i, j, k) = exps[k] * inv_sum;
//                 }
//             } catch (...) {
//                 free(exps);
//                 throw;
//             }
//             free(exps);
//         }
//     }
//     PROFILE_END("softmax_avx2");
// }