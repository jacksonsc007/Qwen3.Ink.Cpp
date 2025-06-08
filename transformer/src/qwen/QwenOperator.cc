#include <cmath>
#include <cstdint>
#include <iomanip>

#include "common.h"
#include "utils.h"
#include "QwenOperator.h"
#include <cassert>
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
    // oss << "[ " << PhaseName << "-qk-" << profile_name << ": (" << bs << ", " << m << ", " << k << ") x (" << bs << ", " << n << ", " << k << ") ]";
    oss << "[ " << PhaseName << "-qk-" << profile_name << "]";
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
    // oss << "[ " << PhaseName << "-pv-" << profile_name << ": (" << bs << ", " << m << ", " << k << ") x (" << bs << ", " << n << ", " << k << ") ]";
    oss << "[ " << PhaseName << "-pv-" << profile_name << "]";
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
    #pragma omp parallel for simd
    for (int i = 0; i < len; i++) {
        float v = a_ptr[i];
        float silu_v = v * (1.0 / (1.0 + exp(-1 * v)));
        output_ptr[i] = silu_v * b_ptr[i];
    }
    PROFILE_END("SiLuMUL");
    return output;
}

Matrix3D<float> add(const Matrix3D<float> a, const Matrix3D<float> b) {
    assert(a.length() == b.length());
    Matrix3D result = a.as_shape();
    for (int i = 0; i < a.length(); i++) {
        result.data()[i] = a.data()[i] + b.data()[i];
    }
    return result;
}

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