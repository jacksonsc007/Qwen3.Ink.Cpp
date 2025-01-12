#include <cmath>
#include <iomanip>

#include "utils.h"
#include "Fp32QwenOperator.h"
#include <cassert>
#include "operators.h"

void QwenRMSNorm::forward(const Matrix3D<float> &x, Matrix3D<float> &output) {
    PROFILE_START(profile_name);
    const int last_dims = 2;

    assert(last_dims == 2);  // support the last dim for now
    assert(output.m_dim_x == x.m_dim_x);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == x.m_dim_z);
    assert(x.m_dim_z == weight.m_dim_z);

    for (int i = 0; i < x.m_dim_x; i++) {      // batches
        for (int j = 0; j < x.m_dim_y; j++) {  // samples
            float var = 0;

            for (int k = 0; k < x.m_dim_z; k++) {  // hideden states
                var += x(i, j, k) * x(i, j, k);
            }
            var /= static_cast<float>(x.m_dim_z);
            float variance = 1.0 / sqrt(var + eps);

            for (int k = 0; k < x.m_dim_z; k++) {
                float value = static_cast<float>(x(i, j, k));
                float fp_out = (value * variance) * weight(0, 0, k);
                output(i, j, k) = fp_out;
            }
        }
    }

    PROFILE_END(profile_name);
}

void Qwen_Linear_FP::forward(const Matrix3D<float> &a, Matrix3D<float> &c) {
    Matrix3D<float> b = this->weight;
    const int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_y == c.m_dim_z);  // n
    // batch dim == 1 only support MM for now
    assert(a.m_dim_x == 1);
    assert(b.m_dim_x == 1);

    struct matmul_params params;
    params.A.row = a.m_dim_y;
    params.A.column = a.m_dim_z;
    params.A.data_ptr = a.m_data;
    params.B.row = b.m_dim_z;     // k
    params.B.column = b.m_dim_y;  // n
    params.B.data_ptr = b.m_data;
    params.C.row = c.m_dim_y;
    params.C.column = c.m_dim_z;
    params.C.data_ptr = c.m_data;
    params.opt_params.blk_size = BLK_SIZE;
    params.opt_params.num_thread = NUM_THREAD;

    matmul::MatmulOperator op = matmul::MatmulOperator();
#if FP32IMP == 0
    op.mat_mul_accelerator_transposed_fastover_column((const struct matmul_params *)&params);
#elif FP32IMP == 1
    op.mat_mul_loop_unrolling4x4_fp32(&params);
#elif FP32IMP == 2
    op.mat_mul_accelerator_transposed_fastover_column_fp32_avx((const struct matmul_params *)&params);
#elif FP32IMP == 3
    op.mat_mul_loop_unrolling4x4_mt_fp32(&params);
#elif FP32IMP == 4
    op.mat_mul_loop_unrolling4x4_mt_avx_fp32(&params);
#elif FP32IMP == 5
    op.mat_mul_loop_unrolling4x4_avx_fp32(&params);
#elif FP32IMP == 6
    op.mat_mul_avx_fp32(&params);
#elif FP32IMP == 7
    op.mat_mul_tiling_fp32(&params);
#elif FP32IMP == 8
    op.mat_mul_loop_unrolling4x4_tiling_fp32(&params);
#elif FP32IMP == 9
    op.mat_mul_multithreading_fp32(&params);
#elif FP32IMP == 10
    op.mat_mul_multithreading_tiling_fp32(&params);
#endif

    // add bias TODO: simd
    Matrix3D<float> bias = this->bias; // (1, n, 1)
    assert (bias.m_dim_y == b.m_dim_y);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c(0, i, j) += bias(0, j, 0);
        }
    }

    PROFILE_END(profile_name);
    return;
}