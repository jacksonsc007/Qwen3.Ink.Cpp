#include <cmath>
#include <iomanip>

#include "utils.h"
#include "Fp32QwenOperator.h"
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
    
void permute01(Matrix3D<float> before, Matrix3D<float> after)
{
    PROFILE_START("Fp32QwenAttention::permute");
    int dim_x = after.m_dim_x;
    int dim_y = after.m_dim_y;
    int dim_z = after.m_dim_z;
    
    for (int i = 0; i < dim_x; i++)
        for (int j = 0; j < dim_y; j++)
            for (int k = 0; k < dim_z; k++)
            {
                // shaped[i, j, k] = unshape[0, j, i * head_dim + k]
                after(i, j, k) = before(j, i, k);
            }
    PROFILE_END("Fp32QwenAttention::permute");
}

// @abstract: reshape a matrix of shape (1, sqlen, embed_dim) to shape (num_head, sqlen, head_dim)
void reshape_headfirst(Matrix3D<float> before, Matrix3D<float> after)
{
    PROFILE_START("Fp32QwenAttention::reshape_headfirst");
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
    PROFILE_END("Fp32QwenAttention::reshape_headfirst");
}

// @abstract: reshape a matrix of shape (num_head, sqlen, head_dim) to (1, sqlen, embed_dim)
void reshape_seqfirst(Matrix3D<float> before, Matrix3D<float> after)
{
    PROFILE_START("Fp32QwenAttention::reshape_seqfirst");
    
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
    PROFILE_END("Fp32QwenAttention::reshape_seqfirst");
}


void Qwen3RMSNorm::forward(const Matrix3D<float> &x, Matrix3D<float> &output, const int dim) {
    PROFILE_START(profile_name);

    assert(dim == -1); // only support apply norm on last dim for now
    // x: (L, n_heads, head_dim)
    // weight: (1, 1, head_dim)
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
