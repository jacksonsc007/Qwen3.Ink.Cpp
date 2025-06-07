#include "QwenOperator.h"
#include "operators.h"
#include "utils.h"

void load_BMM_F32T(BMM_F32T &op, std::string prefix) { 
    read_to_array((prefix + "/alpha.bin").c_str(), &op.alpha, 1); 
}

BMM_F32T::BMM_F32T(float _alpha) { this->alpha = _alpha; }

void BMM_F32T::forward(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output) {
    const int bs = A.m_dim_x;
    const int m = A.m_dim_y, n = B.m_dim_y, k = A.m_dim_z, b_size = B.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: bs x m x k   key_view: bs x n x k   c: bs x m x n
    assert(A.m_dim_x == B.m_dim_x);  // batch dim
    assert(A.m_dim_z == B.m_dim_z);  // k
    assert(A.m_dim_y == output.m_dim_y);  // m
    assert(B.m_dim_y == output.m_dim_z);  // n
    // 
    for (int batch_idx = 0; batch_idx < bs; batch_idx ++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float acc = 0;
                for (int p = 0; p < k; p++)
                {
                    acc += (
                        A(batch_idx, i, p) * B(batch_idx, j, p)
                    );
                }
                output(batch_idx, i, j) = acc * this->alpha;
            }
        }
    }

    // struct matmul_params params;
    // params.A.row = a.m_dim_y;
    // params.A.column = a.m_dim_z;
    // params.A.data_ptr = a.data();
    // params.B.row = b.m_dim_y;
    // params.B.column = b.m_dim_z;
    // params.B.data_ptr = b.data();
    // params.C.row = c.m_dim_y;
    // params.C.column = c.m_dim_z;
    // params.C.data_ptr = c.data();
    // params.opt_params.blk_size = BLK_SIZE;
    // params.opt_params.num_thread = NUM_THREAD;
    // params.alpha = alpha;

    // matmul::MatmulOperator op = matmul::MatmulOperator();

    // for (int bz = 0; bz < a.m_dim_x; bz++) {
    //     // if (params.A.column % 8 == 0) // TODO: debug this
    //     //     op.mat_mul_transposed_fastover_column((const struct matmul_params
    //     //     *)&params);
    //     // else
    //     op.mat_mul_transposed(&params);  // TODO: optimize this
    //     // TODO: apply SIMD here
    //     for (int i = 0; i < m * n; i++) {
    //         params.C.data_ptr[i] *= this->alpha;
    //     }
    //     params.A.data_ptr += m * k;
    //     params.B.data_ptr += k * n;
    //     params.C.data_ptr += m * n;
    // }

    PROFILE_END(profile_name);
}

void BMM_F32T::forward_weight_untransposed(Matrix3D<float> &A, MatrixView<float> &B,
                                           Matrix3D<float> &output) {
    const int m = A.m_dim_y, n = output.m_dim_z, k = A.m_dim_z, bs = B.m_dim_x;
    const long long ops = (long long)bs * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: bs x m x k   b: bs x k x n   c: bs x m x n
    assert(A.m_dim_x == B.m_dim_x);  // batch dim
    assert(A.m_dim_z == B.m_dim_y);  // k
    assert(A.m_dim_y == output.m_dim_y);  // m
    assert(B.m_dim_z == output.m_dim_z);  // n

    // zero out output
    for (int i = 0; i < output.length(); i++)
    {
        output.data()[i] = 0;
    }

    for (int batch_idx = 0; batch_idx < bs; batch_idx ++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int p = 0; p < k; p++)
            {
                for (int j = 0; j < n; j++)
                {
                    output(batch_idx, i, j) += (
                        A(batch_idx, i, p) * B(batch_idx, p, j)
                    );
                }
            }
        }
    }

    // apply alpha
    if (this -> alpha != 1)
    {
        for (int i = 0; i < output.length(); i++)
        {
            output.data()[i] *= this->alpha;
        }
    }

    // struct matmul_params params;
    // params.A.row = A.m_dim_y;
    // params.A.column = A.m_dim_z;
    // params.A.data_ptr = A.data();
    // params.B.row = B.m_dim_y;
    // params.B.column = B.m_dim_z;
    // params.B.data_ptr = B.data();
    // params.C.row = output.m_dim_y;
    // params.C.column = output.m_dim_z;
    // params.C.data_ptr = output.data();
    // params.opt_params.blk_size = BLK_SIZE;
    // params.opt_params.num_thread = NUM_THREAD;
    // params.alpha = alpha;

    // matmul::MatmulOperator op = matmul::MatmulOperator();

    // for (int i = 0; i < m * n * A.m_dim_x; i++) {
    //     params.C.data_ptr[i] = 0;
    // }

    // for (int bz = 0; bz < A.m_dim_x; bz++) {
    //     float *data_A = params.A.data_ptr + bz * m * k, *data_B = params.B.data_ptr + bz * k * n,
    //           *data_C = params.C.data_ptr + bz * m * n;
    //     for (int i = 0; i < m; i++)
    //         for (int kk = 0; kk < k; kk++) {
    //             float Aikk0 = data_A[i * k + kk];
    //             for (int j = 0; j < n; j++) {
    //                 float Bjk0 = data_B[kk * n + j];
    //                 data_C[i * n + j] += Aikk0 * Bjk0;
    //             }
    //         }
    // }

    PROFILE_END(profile_name);
}
