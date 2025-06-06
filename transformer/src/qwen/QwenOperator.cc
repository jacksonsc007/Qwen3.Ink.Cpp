#include <cmath>
#include <cstdint>
#include <iomanip>

#include "common.h"
#include "utils.h"
#include "QwenOperator.h"
#include <cassert>
#include "operators.h"

bool has_nan(Matrix3D<float> mat)
{
    bool res = false;
    int m_dim_x = mat.m_dim_x;
    int m_dim_y = mat.m_dim_y;
    int m_dim_z = mat.m_dim_z;
    for (int i = 0; i < m_dim_x; ++i)
    {
        for (int j = 0; j < m_dim_y; ++j)
        {
            for (int k = 0; k < m_dim_z; ++k)
            {
                float value = mat(i, j, k);
                if (std::isnan(value))
                {
                    return true;
                }
            }
        }
    }
    
    return res;
}
    

// @abstract: reshape a matrix of shape (1, sqlen, embed_dim) to shape (num_head, sqlen, head_dim)
void reshape_headfirst(Matrix3D<float> before, Matrix3D<float> after)
{
    PROFILE_START("QwenAttention::reshape_headfirst");
    int num_heads = after.m_dim_x;
    int sqlen = after.m_dim_y;
    int head_dim = after.m_dim_z;
    
    for (int i = 0; i < num_heads; i++)
        for (int j = 0; j < sqlen; j++)
            for (int k = 0; k < head_dim; k++)
            {
                // shaped[i, j, k] = unshape[0, j, i * head_dim + k]
                after(i, j, k) = before(0, j, i * head_dim + k);
            }
    PROFILE_END("QwenAttention::reshape_headfirst");
}

// @abstract: reshape a matrix of shape (num_head, sqlen, head_dim) to (1, sqlen, embed_dim)
void reshape_seqfirst(Matrix3D<float> before, Matrix3D<float> after)
{
    PROFILE_START("QwenAttention::reshape_seqfirst");
    
    int num_heads = before.m_dim_x;
    int sqlen = before.m_dim_y;
    int head_dim = before.m_dim_z;
    for (int i = 0; i < num_heads; i++)
        for (int j = 0; j < sqlen; j++)
            for (int k = 0; k < head_dim; k++)
            {
               // shape2[0, j, i * head_dim + k] = shape1[i, j, k]
               after(0, j, i * head_dim + k) = before(i, j, k);

            }
    PROFILE_END("QwenAttention::reshape_seqfirst");
}


void Qwen3RMSNorm::load(std::string path)
{
    weight.load(path.c_str());
}


Matrix3D<float> Qwen3RMSNorm::forward(const Matrix3D<float> &x, const int dim) {
    PROFILE_START(profile_name);

    assert(dim == -1); // only support apply norm on last dim for now
    // x: (L, n_heads, head_dim)
    // weight: (1, 1, head_dim)
    assert(x.m_dim_z == weight.m_dim_z);
    Matrix3D<float> output = x.as_shape();

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
    return output;
}


Matrix3D<float> Qwen_Linear_with_bias_Int4::forward(Matrix3D<float> &activation) {
    const int num_thread = 16;
    const int bs = activation.m_dim_x;
    const int m = activation.m_dim_y, n = weight.m_dim_y, k = activation.m_dim_z, b_size = weight.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    Matrix3D<float> output (bs, m, n);
    std::ostringstream oss;
    oss << "[" << profile_name << ": " << m << " x " << n << " x " << k << "]";
    std::string formatted_profile_name = oss.str();
    PROFILE_START_FLOPS(formatted_profile_name, ops);

                                         // A: 1 x m x k float32  B: 1 x n x (k / 2) uint8   C: m x n (float32)
    assert(activation.m_dim_x == weight.m_dim_x);      // batch dim
    assert(activation.m_dim_z / 2 == weight.m_dim_z);  // k
    assert(activation.m_dim_y == output.m_dim_y);      // m
    assert(weight.m_dim_y == output.m_dim_z);      // n
                                         // batch dim == 1 only support MM for now
    assert(activation.m_dim_x == 1);
    assert(weight.m_dim_x == 1);

    struct qwen_matmul_params params;
    params.A.row                 = activation.m_dim_y;
    params.A.column              = activation.m_dim_z;
    params.A.data_ptr            = activation.data();
    params.B.row                 = weight.m_dim_z;                // k
    params.B.column              = weight.m_dim_y;                // n
    params.B.int4_data_ptr       = weight.data();

    params.C.row                 = output.m_dim_y;
    params.C.column              = output.m_dim_z;
    params.C.data_ptr            = output.data();
    params.opt_params.num_thread = NUM_THREAD;
    params.scales                = this->scale.data();
    params.offset                = this->offset.data();
    params.zero_point            = this->zero_point.data();
    params.block_size            = QK;

    activation_int8 = Matrix3D<int8_t>(bs, m, k);
    activation_scale = Matrix3D<float>(bs, m, k / QK);
    params.A.int8_data_ptr = activation_int8.data();
    params.A_scales = activation_scale.data();

    matmul::MatmulOperator op = matmul::MatmulOperator();
    // op.matMul_int4_multiThread_qwen(&params);
    // op.matMul_int4_avx_qwen(&params);
    // op.matMul_int4_multiThread_avx_qwen(&params);
    // op.matMul_int4_Tiling1vl_qwen(&params);
    // op.matMul_int4_unrolling2x2_qwen(&params);
    // op.matMul_int4Reference_qwen(&params);
    // op.matMul_int4_multiThread_avx_qwen(&params);
    
    if (m != 1)
    {
        op.qgemm_A80W40_kernel(&params);
    }
    else 
    {
        op.qgemv_A80W40_kernel(&params);
    }


    // add bias TODO: simd
    if (has_bias)
    {
        std::ostringstream oss;
        oss << "[" << profile_name << " bias_add: " << m << " x " << n << " x " << k << "]";
        std::string formatted_profile_name = oss.str();
        Matrix3D<float> bias = this->bias; // (1, n, 1)
        assert (bias.m_dim_y == weight.m_dim_y);
        PROFILE_START(formatted_profile_name);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                output(0, i, j) += bias(0, j, 0);
            }
        }
        PROFILE_END(formatted_profile_name);
    }

    PROFILE_END(formatted_profile_name);
    return output;
}