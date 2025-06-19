#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>

#include "common.h"
#include "utils.h"
#include "QwenOperator.h"
#include <cassert>
#include <iostream>
#include "operators.h"
#include <blis/cblas.h>
#include <cblas.h>

    

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


Matrix3D<float> Qwen_Linear_with_bias_Int4::forward_gemm(const Matrix3D<float> &activation) {
    const int bs = activation.m_dim_x;
    const int m = activation.m_dim_y, n = weight_cols, k = activation.m_dim_z, b_size = activation.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START("[" + profile_name + " ::" + "create output]");
    Matrix3D<float> output (bs, m, n);
    PROFILE_END("[" + profile_name + " ::" + "create output]");
    std::ostringstream oss;
    oss << "[" << profile_name << ": " << m << " x " << n << " x " << k << "]";
    std::string formatted_profile_name = oss.str();
    PROFILE_START_FLOPS(formatted_profile_name, ops);



    int8_t * A_repack = context_->activation_buffer.get();
    PROFILE_START("[" + profile_name + " ::" + "Activation Online Quantization]");
    quantize_row_q8_1_repack(activation.data(), A_repack, m * k);
    PROFILE_END("[" + profile_name + " ::" + "Activation Online Quantization]");

    PROFILE_START_FLOPS(formatted_profile_name + " ::" + "real computaion", ops);
    gemm_repack_A81W41(
        A_repack, weight_repack.get(), output.data(),
        m, n, k
    );
    PROFILE_END(formatted_profile_name + " ::" + "real computaion");
    gemv_repack_A81W41(
        A_repack, weight_repack.get(), output.data(),
        m, n, k
    );

    PROFILE_END(formatted_profile_name);
    return output;
}


Matrix3D<float> Qwen_Linear_with_bias_Int4::forward(const Matrix3D<float> &activation) {
    const int bs = activation.m_dim_x;
    const int m = activation.m_dim_y, n = weight_cols, k = activation.m_dim_z, b_size = activation.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START("[" + profile_name + " ::" + "create output]");
    Matrix3D<float> output (bs, m, n);
    PROFILE_END("[" + profile_name + " ::" + "create output]");
    std::ostringstream oss;
    oss << "[" << profile_name << ": " << m << " x " << n << " x " << k << "]";
    std::string formatted_profile_name = oss.str();
    PROFILE_START_FLOPS(formatted_profile_name, ops);



    int8_t * A_repack = context_->activation_buffer.get();
    PROFILE_START("[" + profile_name + " ::" + "Activation Online Quantization]");
    quantize_row_q8_1_repack(activation.data(), A_repack, m * k);
    PROFILE_END("[" + profile_name + " ::" + "Activation Online Quantization]");

    if (m > 1)
    {

        PROFILE_START_FLOPS(formatted_profile_name + " ::" + "real computaion", ops);
        gemm_repack_A81W41(
            A_repack, weight_repack.get(), output.data(),
            m, n, k
        );
        PROFILE_END(formatted_profile_name + " ::" + "real computaion");
    }
    else
    {
        gemv_repack_A81W41(
            A_repack, weight_repack.get(), output.data(),
            m, n, k
        );
    }

    PROFILE_END(formatted_profile_name);
    return output;
}

void load_BMM_F32T(bgemmGQA &op, std::string prefix) { 
    read_to_array((prefix + "/alpha.bin").c_str(), &op.alpha, 1); 
}

bgemmGQA::bgemmGQA(float _alpha, int num_q_head, int num_kv_head) { 
    this->alpha = _alpha; 
    groupsize = num_q_head / num_kv_head;
}

void bgemmGQA::forward(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output) {
    const int m = A.m_dim_y, n = B.m_dim_y, k = A.m_dim_z, bs = A.m_dim_x;
    const long long ops = (long long)bs * 2 * (long long)m * (long long)n * (long long)k;
    std::ostringstream oss;
    std::string PhaseName;
    if (m > 1)
    {
        PhaseName = "P Stage";
    }
    else {
        PhaseName = "AG Stage";
    }
    oss << "[ " << PhaseName << "-qk-" << profile_name << ": (" << bs << ", " << m << ", " << k << ") x (" << bs << ", " << n << ", " << k << ") ]";
    // oss << "[ " << PhaseName << "-qk-" << profile_name << "]";
    std::string formatted_profile_name = oss.str();
    PROFILE_START_FLOPS(formatted_profile_name, ops);

    // a: bs x m x k   key_view: bs x n x k   c: bs x m x n
    assert(A.m_dim_x == groupsize * B.m_dim_x);  // batch dim
    assert(A.m_dim_z == B.m_dim_z);  // k
    assert(A.m_dim_y == output.m_dim_y);  // m
    assert(B.m_dim_y == output.m_dim_z);  // n
    // 
    int n_heads = bs;
    for (int head_idx = 0; head_idx < n_heads; head_idx ++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float acc = 0;
                for (int p = 0; p < k; p++)
                {
                    acc += (
                        A(head_idx, i, p) * B(head_idx / groupsize, j, p)
                    );
                }
                output(head_idx, i, j) = acc * this->alpha;
            }
        }
    }

    PROFILE_END(formatted_profile_name);
}


void bgemmGQA::forward_ink_kernel_qk(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output)
{
    int bs = A.m_dim_x;
    int m = A.m_dim_y;
    int n = B.m_dim_y;
    int k = A.m_dim_z;
    const int groupsize = this->groupsize;

    // Total FLOPS for profiling
    const long long ops = (long long)bs * 2 * (long long)m * (long long)n * (long long)k;
    std::ostringstream oss;
    std::string PhaseName;
    if (m > 1) {
        PhaseName = "P Stage";
    } else {
        PhaseName = "AG Stage";
    }
    // oss << "[ " << PhaseName << "-qk-" << profile_name << ": (" << bs << ", " << m << ", " << k << ") x (" << bs << ", " << n << ", " << k << ") ]";
    oss << "[ " << PhaseName << "-qk-" << profile_name << "]";
    std::string formatted_profile_name = oss.str();
    PROFILE_START_FLOPS(formatted_profile_name, ops);

    // a: bs x m x k   key_view: bs x n x k   c: bs x m x n
    int n_heads = bs;
    if ( m > 1)
    {
        for (int head_idx = 0; head_idx < n_heads; head_idx ++)
        {
            gemm_fp32_rcr(
                A.data() + head_idx * m * k,  // A: bs x m x k
                &B(head_idx / groupsize, 0, 0),  // B: bs/groupsize x n x k
                output.data() + head_idx * m * n,  // C: bs x m x n
                m, n, k
            );
        }
    }
    else {
        for (int head_idx = 0; head_idx < n_heads; head_idx ++)
        {
            gemv_fp32_rcr_mt_impl_2(
                A.data() + head_idx * m * k,  // A: bs x m x k
                &B(head_idx / groupsize, 0, 0),  // B: bs/groupsize x n x k
                output.data() + head_idx * m * n,  // C: bs x m x n
                m, n, k
            );
        }
    }
    PROFILE_START("apply scaling factor");
    for (int head_idx = 0; head_idx < n_heads; head_idx ++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                output(head_idx, i, j) = output(head_idx, i, j) * this->alpha;
            }
        }
    }
    PROFILE_END("apply scaling factor");


    PROFILE_END(formatted_profile_name);
}

void bgemmGQA::forward_ink_kernel_pv(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output) {
    const int m = A.m_dim_y, n = output.m_dim_z, k = A.m_dim_z, bs = A.m_dim_x;
    const long long ops = (long long)bs * 2 * (long long)m * (long long)n * (long long)k;
    std::ostringstream oss;
    std::string PhaseName;
    if (m > 1)
    {
        PhaseName = "P Stage";
    }
    else {
        PhaseName = "AG Stage";
    }
    // oss << "[ " << PhaseName << "-pv-" << profile_name << ": (" << bs << ", " << m << ", " << k << ") x (" << bs << ", " << n << ", " << k << ") ]";
    oss << "[ " << PhaseName << "-pv-" << profile_name << "]";
    std::string formatted_profile_name = oss.str();
    PROFILE_START_FLOPS(formatted_profile_name, ops);

    // a: bs x m x k   b: bs x k x n   c: bs x m x n
    assert(A.m_dim_x == groupsize * B.m_dim_x);  // batch dim
    assert(A.m_dim_z == B.m_dim_y);  // k
    assert(A.m_dim_y == output.m_dim_y);  // m
    assert(B.m_dim_z == output.m_dim_z);  // n

    // zero out output
    for (int i = 0; i < output.length(); i++)
    {
        output.data()[i] = 0;
    }

    int n_heads = bs;
    if (m > 1)
    {
        for (int head_idx = 0; head_idx < n_heads; head_idx ++)
        {
            gemm_fp32_rrr(
                A.data() + head_idx * m * k,  // A: bs x m x k
                &B(head_idx / groupsize, 0, 0),  // B: bs/groupsize x n x k
                output.data() + head_idx * m * n,  // C: bs x m x n
                m, n, k
            );
        }
    }
    else{
        for (int head_idx = 0; head_idx < n_heads; head_idx ++)
        {
            // we find multi-threading does not help.
            gemv_fp32_rrr_naive(
                A.data() + head_idx * m * k,  // A: bs x m x k
                &B(head_idx / groupsize, 0, 0),  // B: bs/groupsize x n x k
                output.data() + head_idx * m * n,  // C: bs x m x n
                m, n, k
            );
        }

    }

    PROFILE_END(formatted_profile_name);
}



#ifdef OPENBLAS

void bgemmGQA::forward_mix_kernel_pv(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output)
{

    const int m = A.m_dim_y, n = output.m_dim_z, k = A.m_dim_z, bs = A.m_dim_x;
    const long long ops = (long long)bs * 2 * (long long)m * (long long)n * (long long)k;
    std::ostringstream oss;
    std::string PhaseName;
    if (m > 1)
    {
        PhaseName = "P Stage";
    }
    else {
        PhaseName = "AG Stage";
    }
    oss << "[ " << PhaseName << "-pv-mixed_kernel" << profile_name << ": (" << bs << ", " << m << ", " << k << ") x (" << bs << ", " << n << ", " << k << ") ]";
    // oss << "[ " << PhaseName << "-pv-" << profile_name << "]";
    std::string formatted_profile_name = oss.str();
    PROFILE_START_FLOPS(formatted_profile_name, ops);

    // a: bs x m x k   b: bs x k x n   c: bs x m x n
    assert(A.m_dim_x == groupsize * B.m_dim_x);  // batch dim
    assert(A.m_dim_z == B.m_dim_y);  // k
    assert(A.m_dim_y == output.m_dim_y);  // m
    assert(B.m_dim_z == output.m_dim_z);  // n

    // zero out output
    for (int i = 0; i < output.length(); i++)
    {
        output.data()[i] = 0;
    }

    // Prepare array of pointers to A, B, and C matrices
    float **A_pointers = new float*[bs];
    float **B_pointers = new float*[bs];
    float **C_pointers = new float*[bs];
    // Populate pointers
    for (int i = 0; i < bs; ++i) {
        A_pointers[i] = A.data() + i * m * k;

        // Each group shares B matrices
        int b_index = i / groupsize;
        // NOTE: memory layout is not contiguous inside matrixview
        // B_pointers[i] = B.data() + b_index * n * k;
        B_pointers[i] = &B(b_index, 0, 0);

        C_pointers[i] = output.data() + i * m * n;
    }

    const float alpha = 1;
    const float beta = 0.0f;  // We don't reuse output
    int n_heads = bs;
    if (m > 1)
    {
        for (int head_idx = 0; head_idx < n_heads; head_idx ++)
        {
            gemm_fp32_rrr(
                A_pointers[head_idx],
                B_pointers[head_idx],
                C_pointers[head_idx],
                m, n, k
            );
        }
    }
    else{
        // Loop over all batchesm * k
        // A: (bs, m=1, k) = (bs, 1, k)
        // B: row-major (bs, k, n); View as column-major (bs, n, k)
        for (int i = 0; i < bs; ++i) {
            float* a = A_pointers[i];
            float* b_matrix = B_pointers[i];
            // Output scalar for this batch
            float* c = C_pointers[i];

            // Perform dot product between A[i] and each row of B[b_index]
            // This is equivalent to: cblas_sgemv with transposed B
            cblas_sgemv( 
                CblasColMajor,
                CblasNoTrans,      // B^T * A
                n,                    // Number of rows in B
                k,                    // Number of cols in B
                alpha,             // alpha
                b_matrix,             // B matrix
                n,                  // leading dimension of B (cols)  
                a,                    // A vector (length n)
                1,                 // stride
                beta,              // beta
                c,                    // result scalar
                1                  // stride
            );
        }
    }

    PROFILE_END(formatted_profile_name);
}

void bgemmGQA::forward_openblas_qk(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output) {
    int bs = A.m_dim_x;
    int m = A.m_dim_y;
    int n = B.m_dim_y;
    int k = A.m_dim_z;
    const int groupsize = this->groupsize;

    // Total FLOPS for profiling
    const long long ops = (long long)bs * 2 * (long long)m * (long long)n * (long long)k;
    std::ostringstream oss;
    std::string PhaseName;
    if (m > 1) {
        PhaseName = "P Stage";
    } else {
        PhaseName = "AG Stage";
    }
    oss << "[ " << PhaseName << "-qk-" << profile_name << ": (" << bs << ", " << m << ", " << k << ") x (" << bs << ", " << n << ", " << k << ") ]";
    // oss << "[ " << PhaseName << "-qk-" << profile_name << "]";
    std::string formatted_profile_name = oss.str();
    PROFILE_START_FLOPS(formatted_profile_name, ops);

    // Setup arrays for batched GEMM
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasTrans;  // B is of shape (n, k), so transpose to (k x n)

    const float alpha = this->alpha;
    const float beta = 0.0f;  // We don't reuse output

    // Prepare array of pointers to A, B, and C matrices
    float **A_pointers = new float*[bs];
    float **B_pointers = new float*[bs];
    float **C_pointers = new float*[bs];

    int lda = k;  // A is m x k, row-major
    int ldb = k;  // B is n x k, row-major, but we transpose it
    int ldc = n;  // C is m x n, row-major

    // Populate pointers
    for (int i = 0; i < bs; ++i) {
        A_pointers[i] = A.data() + i * m * k;

        // Each group shares B matrices
        int b_index = i / groupsize;
        // NOTE: memory layout is not contiguous inside matrixview
        // B_pointers[i] = B.data() + b_index * n * k;
        B_pointers[i] = &B(b_index, 0, 0);

        C_pointers[i] = output.data() + i * m * n;
    }
    if (m > 1)
    {
        // Batched GEMM call
        cblas_sgemm_batch(
            CblasRowMajor,
            &transA, &transB,
            &m, &n, &k,
            &alpha,
            (const float**)A_pointers, &lda,
            (const float**)B_pointers, &ldb,
            &beta,
            C_pointers, &ldc,
            1, &bs
        );
    }
    else
    {
        // Loop over all batches
        // A: (bs, 1, k) or (bs, k, 1)
        // B: (bs, n, k)
        // Perform gemv of B * A
        for (int i = 0; i < bs; ++i) {
            float* a = A_pointers[i];
            float* b_matrix = B_pointers[i];
            // Output scalar for this batch
            float* c = C_pointers[i];

            // Perform dot product between A[i] and each row of B[b_index]
            // This is equivalent to: cblas_sgemv with transposed B
            cblas_sgemv( 
                CblasRowMajor,
                CblasNoTrans,      // B^T * A
                n,                    // Number of rows in B
                k,                    // Number of cols in B
                alpha,             // alpha
                b_matrix,             // B matrix
                k,                  // leading dimension of B (cols)  
                a,                    // A vector (length n)
                1,                 // stride
                beta,              // beta
                c,                    // result scalar
                1                  // stride
            );
        }
    }

    delete[] A_pointers;
    delete[] B_pointers;
    delete[] C_pointers;

    PROFILE_END(formatted_profile_name);
}

void bgemmGQA::forward_openblas_pv(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output) {
    int bs = A.m_dim_x;
    int m = A.m_dim_y;
    int k = A.m_dim_z;
    int n = B.m_dim_z;
    const int groupsize = this->groupsize;

    // Total FLOPS for profiling
    const long long ops = (long long)bs * 2 * (long long)m * (long long)n * (long long)k;
    std::ostringstream oss;
    std::string PhaseName;
    if (m > 1) {
        PhaseName = "P Stage";
    } else {
        PhaseName = "AG Stage";
    }
    oss << "[ " << PhaseName << "-pv-" << profile_name << ": (" << bs << ", " << m << ", " << k << ") x (" << bs << ", " << n << ", " << k << ") ]";
    // oss << "[ " << PhaseName << "-pv-" << profile_name << "]";
    std::string formatted_profile_name = oss.str();
    PROFILE_START_FLOPS(formatted_profile_name, ops);

    // Setup arrays for batched GEMM
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;  // B is of shape (k, n), so No need to transpose

    const float alpha = 1;
    const float beta = 0.0f;  // We don't reuse output

    // Prepare array of pointers to A, B, and C matrices
    float **A_pointers = new float*[bs];
    float **B_pointers = new float*[bs];
    float **C_pointers = new float*[bs];
    // Populate pointers
    for (int i = 0; i < bs; ++i) {
        A_pointers[i] = A.data() + i * m * k;

        // Each group shares B matrices
        int b_index = i / groupsize;
        // NOTE: memory layout is not contiguous inside matrixview
        // B_pointers[i] = B.data() + b_index * n * k;
        B_pointers[i] = &B(b_index, 0, 0);

        C_pointers[i] = output.data() + i * m * n;
    }
    // batched-gemm
    if (m > 1)
    {

        int lda = k;  // A is m x k, row-major
        int ldb = n;  // B is k x n, row-major
        int ldc = n;  // C is m x n, row-major
        // Batched GEMM call
        cblas_sgemm_batch(
            CblasRowMajor,
            &transA, &transB,
            &m, &n, &k,
            &alpha,
            (const float**)A_pointers, &lda,
            (const float**)B_pointers, &ldb,
            &beta,
            C_pointers, &ldc,
            1, &bs
        );
    }
    // batched-gemv
    else{
        // Loop over all batchesm * k
        // A: (bs, m=1, k) = (bs, 1, k)
        // B: row-major (bs, k, n); View as column-major (bs, n, k)
        for (int i = 0; i < bs; ++i) {
            float* a = A_pointers[i];
            float* b_matrix = B_pointers[i];
            // Output scalar for this batch
            float* c = C_pointers[i];

            // Perform dot product between A[i] and each row of B[b_index]
            // This is equivalent to: cblas_sgemv with transposed B
            cblas_sgemv( 
                CblasColMajor,
                CblasNoTrans,      // B^T * A
                n,                    // Number of rows in B
                k,                    // Number of cols in B
                alpha,             // alpha
                b_matrix,             // B matrix
                n,                  // leading dimension of B (cols)  
                a,                    // A vector (length n)
                1,                 // stride
                beta,              // beta
                c,                    // result scalar
                1                  // stride
            );
        }
    }
    delete[] A_pointers;
    delete[] B_pointers;
    delete[] C_pointers;
    PROFILE_END(formatted_profile_name);
}
#endif

void bgemmGQA::forward_weight_untransposed(Matrix3D<float> &A, MatrixView<float> &B,
                                           Matrix3D<float> &output) {
    const int m = A.m_dim_y, n = output.m_dim_z, k = A.m_dim_z, bs = A.m_dim_x;
    const long long ops = (long long)bs * 2 * (long long)m * (long long)n * (long long)k;
    std::ostringstream oss;
    std::string PhaseName;
    if (m > 1)
    {
        PhaseName = "P Stage";
    }
    else {
        PhaseName = "AG Stage";
    }
    oss << "[ " << PhaseName << "-pv-" << profile_name << ": (" << bs << ", " << m << ", " << k << ") x (" << bs << ", " << n << ", " << k << ") ]";
    // oss << "[ " << PhaseName << "-pv-" << profile_name << "]";
    std::string formatted_profile_name = oss.str();
    PROFILE_START_FLOPS(formatted_profile_name, ops);

    // a: bs x m x k   b: bs x k x n   c: bs x m x n
    assert(A.m_dim_x == groupsize * B.m_dim_x);  // batch dim
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
                        A(batch_idx, i, p) * B(batch_idx / groupsize, p, j)
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
    PROFILE_END(formatted_profile_name);
}

Matrix3D<float> Qwen3SiLuMul(const Matrix3D<float> &a, const Matrix3D<float> &b) {
    PROFILE_START("SiLuMUL");
    Matrix3D<float> output = a.as_shape();
    int len = a.length();
    float * output_ptr = output.data();
    const float * a_ptr = a.data();
    const float * b_ptr = b.data();
    #pragma omp parallel for simd num_threads(16)
    for (int i = 0; i < len; i++) {
        float v = a_ptr[i];
        float silu_v = v * (1.0 / (1.0 + exp(-1 * v)));
        output_ptr[i] = silu_v * b_ptr[i];
    }
    PROFILE_END("SiLuMUL");
    return output;
}

Matrix3D<float> add(Matrix3D<float> &a, Matrix3D<float> &b) {
    assert(a.length() == b.length());
    Matrix3D result = a.as_shape();
    for (int i = 0; i < a.length(); i++) {
        result.data()[i] = a.data()[i] + b.data()[i];
    }
    return result;
}

bool has_nan(Matrix3D<float> & mat)
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



#define A(i, j, ld) ( A + ( i ) * ( ld ) + ( j ) )
#define B(i, j, ld) ( B + ( j ) * ( ld ) + ( i ) )
#define C(i, j, ld) ( C + ( i ) * ( ld ) + ( j ) )
#define SA(i, j, ld) (SA + (i) * (ld) + (j))
#define SB(i, j, ld) (SB + (j) * (ld) + (i))
#define MinB(i, j, ld) (MinB + (j) * (ld) + (i))
#define ScaledSumA(i, j, ld) (ScaledSumA + (i) * (ld) + (j))

void Qwen_Linear_with_bias_Int4::repack_w80_weight(const int K, const int N, const int Q_BLK_SIZE, void * B_repack, 
    const float * SB, const uint8_t * B)
    {
        
    int num_repack_blk_B_along_K = K / (2 * Q_BLK_SIZE);
    int num_repack_blk_B_along_N = N / 8;
    struct q4_repack_2x8 * B_start =  (struct q4_repack_2x8 *) B_repack;
    for (int j = 0; j < num_repack_blk_B_along_N; j++) 
    {
        for (int i = 0; i < num_repack_blk_B_along_K; i++)
        {
            struct q4_repack_2x8 * B_ptr =  B_start + j * num_repack_blk_B_along_K + i;

            // pack scaling factors
            for (int jj = 0; jj < 8; jj++)
            {
                // Convert FP32 to FP16 to improve gemv performance
                fp16_t s_low  = GGML_FP32_TO_FP16( *SB ( i * 2    , j * 8 + jj, K / Q_BLK_SIZE ) ); 
                fp16_t s_high = GGML_FP32_TO_FP16( *SB ( i * 2 + 1, j * 8 + jj, K / Q_BLK_SIZE ) ); 
                B_ptr->s_low[jj] = s_low;
                B_ptr->s_high[jj] = s_high;
            }
            // pack quantized int4 weights in an interleaved manner
            // The block has size of 64 x 8 elements logically, but physically it is laid out as 32 x 8,
            // since each element has 4 bits and the block has type int8, wiht each element only taking half of the byte.
            // 32 x 8 is covered by stacking 8 packs, each measuring 4 x 8 elements.
            // memory layout inside packs is in column-major order
            uint8_t * B_ptr_start =  B_ptr->q_coupled;
            int num_packs = 8;
            for (int pack_idx = 0; pack_idx < num_packs; pack_idx++)
            {
                uint8_t * pack_start_ptr = B_ptr_start + pack_idx * 32;
                for (int jj = 0; jj < 8; jj++)
                {
                    for (int ii = 0; ii < 4; ii++)
                    {
                        pack_start_ptr[ii + jj * 4] = *B(i * Q_BLK_SIZE + pack_idx * 4 + ii, j * 8 + jj, K / 2);
                    }
                }
            }
        }
    } 
    }

void Qwen_Linear_with_bias_Int4::repack_w81_weight(const int K, const int N, const int Q_BLK_SIZE, void * B_repack, 
    const float * SB, const float * MinB, const uint8_t * B)
    {

    int num_repack_blk_B_along_K = K / (2 * Q_BLK_SIZE);
    int num_repack_blk_B_along_N = N / 8;
    struct q4_repack_2x8 * B_start =  (struct q4_repack_2x8 *) B_repack;
    for (int j = 0; j < num_repack_blk_B_along_N; j++) 
    {
        for (int i = 0; i < num_repack_blk_B_along_K; i++)
        {
            struct q4_repack_2x8 * B_ptr =  B_start + j * num_repack_blk_B_along_K + i;

            // pack scaling factors
            for (int jj = 0; jj < 8; jj++)
            {
                float s_low_fp32 = *SB   ( i * 2    , j * 8 + jj, K / Q_BLK_SIZE ); 
                float s_high_fp32 = *SB   ( i * 2 + 1, j * 8 + jj, K / Q_BLK_SIZE ); 
                float min_low_fp32 = *MinB ( i * 2    , j * 8 + jj, K / Q_BLK_SIZE ); 
                float min_high_fp32 = *MinB ( i * 2 + 1, j * 8 + jj, K / Q_BLK_SIZE ); 
                fp16_t s_low    = GGML_FP32_TO_FP16(s_low_fp32); 
                fp16_t s_high   = GGML_FP32_TO_FP16(s_high_fp32); 
                fp16_t min_low  = GGML_FP32_TO_FP16(min_low_fp32);
                fp16_t min_high = GGML_FP32_TO_FP16(min_high_fp32);
                B_ptr->s_low[jj] = s_low;
                B_ptr->s_high[jj] = s_high;
                B_ptr->min_low[jj] = min_low;
                B_ptr->min_high[jj] = min_high;
            }
            // pack quantized int4 weights in an interleaved manner
            // The block has size of 64 x 8 elements logically, but physically it is laid out as 32 x 8,
            // since each element has 4 bits and the block has type int8, wiht each element only taking half of the byte.
            // 32 x 8 is covered by stacking 8 packs, each measuring 4 x 8 elements.
            // memory layout inside packs is in column-major order
            uint8_t * B_ptr_start =  B_ptr->q_coupled;
            int num_packs = 8;
            for (int pack_idx = 0; pack_idx < num_packs; pack_idx++)
            {
                uint8_t * pack_start_ptr = B_ptr_start + pack_idx * 32;
                for (int jj = 0; jj < 8; jj++)
                {
                    for (int ii = 0; ii < 4; ii++)
                    {
                        pack_start_ptr[ii + jj * 4] = *B(i * Q_BLK_SIZE + pack_idx * 4 + ii, j * 8 + jj, K / 2);
                    }
                }
            }
        }
    } 
    }