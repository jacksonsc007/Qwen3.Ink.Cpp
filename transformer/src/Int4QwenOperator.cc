#include <cmath>
#include <iomanip>

#include "utils.h"
#include "Int4QwenOperator.h"
#include <cassert>
#include "operators.h"

void QwenInt4RMSNorm::forward(const Matrix3D<float> &x, Matrix3D<float> &output) 
{
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
/*
x_int8 : quantized int8 actication
x_scale: scaling factors
*/
static int8_t *x_int8;
static float *x_scale;

void Qwen_Linear_with_bias_Int4::initialize_memory(const int block_size)
{
    // reserve memory for maximum sequence length (1, max_sqlen, embed_dim)
    // ques: is this an error? MAX_LINEAR_LENGTH
    allocate_aligned_memory(x_int8, MAX_LINEAR_LENGTH * sizeof(int8_t));
    allocate_aligned_memory(x_scale, (MAX_LINEAR_LENGTH / block_size) * sizeof(float) );
}

void Qwen_Linear_with_bias_Int4::forward(const Matrix3D<float> &a, Matrix3D<float> &c) {
    const int num_thread = 16;
    Matrix3D<uint8_t> b = this->weight;
    const int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

                                         // A: 1 x m x k float32  B: 1 x n x (k / 2) uint8   C: m x n (float32)
    assert(a.m_dim_x == b.m_dim_x);      // batch dim
    assert(a.m_dim_z / 2 == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);      // m
    assert(b.m_dim_y == c.m_dim_z);      // n
                                         // batch dim == 1 only support MM for now
    assert(a.m_dim_x == 1);
    assert(b.m_dim_x == 1);

    struct qwen_matmul_params params;
    params.A.row                 = a.m_dim_y;
    params.A.column              = a.m_dim_z;
    params.A.data_ptr            = a.m_data;
    params.B.row                 = b.m_dim_z;                // k
    params.B.column              = b.m_dim_y;                // n
    params.B.int4_data_ptr       = b.m_data;

    params.B.data_ptr       = this->fp32_weight.m_data;

    params.C.row                 = c.m_dim_y;
    params.C.column              = c.m_dim_z;
    params.C.data_ptr            = c.m_data;
    params.opt_params.num_thread = NUM_THREAD;
    params.scales                = this->scale.m_data;
    params.offset                = this->offset.m_data;
    params.zero_point            = this->zero_point.m_data;
    params.block_size            = QK;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    if (!x_int8) this->initialize_memory(QK);
    params.A.int8_data_ptr = x_int8;
    params.A_scales = x_scale;
    // op.matMul_int4_multiThread_qwen(&params);
    // op.matMul_int4_avx_qwen(&params);
    op.matMul_int4_multiThread_avx_qwen(&params);
    // op.matMul_int4_Tiling1vl_qwen(&params);
    // op.matMul_int4_unrolling2x2_qwen(&params);
    // op.matMul_int4Reference_qwen(&params);
    


    // add bias TODO: simd
    if (has_bias)
    {
        Matrix3D<float> bias = this->bias; // (1, n, 1)
        assert (bias.m_dim_y == b.m_dim_y);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                c(0, i, j) += bias(0, j, 0);
            }
        }
    }

    PROFILE_END(profile_name);
    return;
}

void Qwen_Linear_with_bias_Int4::forward_reference(const Matrix3D<float> &a, Matrix3D<float> &c) {
    const int num_thread = 16;
    Matrix3D<uint8_t> b = this->weight;
    const int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    // PROFILE_START_FLOPS(profile_name, ops);

                                         // A: 1 x m x k float32  B: 1 x n x (k / 2) uint8   C: m x n (float32)
    assert(a.m_dim_x == b.m_dim_x);      // batch dim
    assert(a.m_dim_z / 2 == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);      // m
    assert(b.m_dim_y == c.m_dim_z);      // n
                                         // batch dim == 1 only support MM for now
    assert(a.m_dim_x == 1);
    assert(b.m_dim_x == 1);

    struct qwen_matmul_params params;
    params.A.row                 = a.m_dim_y;
    params.A.column              = a.m_dim_z;
    params.A.data_ptr            = a.m_data;
    params.B.row                 = b.m_dim_z;                // k
    params.B.column              = b.m_dim_y;                // n
    params.B.int4_data_ptr       = b.m_data;

    params.B.data_ptr       = this->fp32_weight.m_data;

    params.C.row                 = c.m_dim_y;
    params.C.column              = c.m_dim_z;
    params.C.data_ptr            = c.m_data;
    params.opt_params.num_thread = NUM_THREAD;
    params.scales                = this->scale.m_data;
    params.offset                = this->offset.m_data;
    params.zero_point            = this->zero_point.m_data;
    params.block_size            = QK;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    if (!x_int8) this->initialize_memory(QK);
    params.A.int8_data_ptr = x_int8;
    params.A_scales = x_scale;
    op.matMul_int4Reference_qwen(&params);
    // add bias TODO: simd
    if (has_bias)
    {
        Matrix3D<float> bias = this->bias; // (1, n, 1)
        assert (bias.m_dim_y == b.m_dim_y);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                c(0, i, j) += bias(0, j, 0);
            }
        }
    }

    // PROFILE_END(profile_name);
    return;
}