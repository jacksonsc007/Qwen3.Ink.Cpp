#ifndef _QWENOPERATOR_H
#define _QWENOPERATOR_H

#include <cstddef>
#include <cstdint>
#include "common.h"
#include "utils.h"
#include <cassert>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include "QwenConfig.h"
#include "ggml-impl.h"





// Helper to create aligned unique_ptr
template <typename T>
std::unique_ptr<T, void(*)(T*)> make_aligned_x86(size_t align, size_t count) {
    // Ensure alignment is a power of two
    if (align & (align - 1)) {
        throw std::invalid_argument("Alignment must be a power of two");
    }
    auto deleter = [](T* ptr) { _mm_free(ptr); };
    T* ptr = static_cast<T*>(_mm_malloc(count * sizeof(T), align));
    if (!ptr && count > 0) {
        throw std::bad_alloc();
    }
    return {ptr, deleter};
}

struct ModelContext{
    std::unique_ptr<float[]> k_cache;
    std::unique_ptr<float[]> v_cache;
    std::unique_ptr<int8_t, void(*)(int8_t *)> repack_buffer; // buffer to load weights before repacking
    std::unique_ptr<int8_t, void(*)(int8_t *)> activation_buffer; // buffer to load online-quantized activations in linear layers 
    
    // Null deleter for empty states
    static void null_deleter(int8_t*) {}
    ModelContext():repack_buffer(nullptr, null_deleter), activation_buffer(nullptr, null_deleter){}
};


typedef uint16_t fp16_t;

struct q4_repack_2x8_fp16{
    fp16_t s_low[8];
    fp16_t s_high[8];
    fp16_t min_low[8];
    fp16_t min_high[8];
    uint8_t q_coupled[256]; // (64 * 8) / (8 / 4)
};

struct q8_repack_1x2_fp16{
    fp16_t s_low;
    fp16_t s_high;
    fp16_t scaled_sum_low;
    fp16_t scaled_sum_high;
    int8_t q_low[32];
    int8_t q_high[32];
};

struct q4_repack_2x8_fp32{
    float s_low[8];
    float s_high[8];
    float min_low[8];
    float min_high[8];
    uint8_t q_coupled[256]; // (64 * 8) / (8 / 4)
};

struct q8_repack_1x2_fp32{
    float s_low;
    float s_high;
    float scaled_sum_low;
    float scaled_sum_high;
    int8_t q_low[32];
    int8_t q_high[32];
};

void repack_w80_weight(const int K, const int N, const int Q_BLK_SIZE, void * B_repack, const float * SB, const uint8_t * B);
void repack_w81_weight_fp16(const int K, const int N, const int Q_BLK_SIZE, void * B_repack, const float * SB, const float * MinB, const uint8_t * B);
void repack_w81_weight_fp32(const int K, const int N, const int Q_BLK_SIZE, void * B_repack, const float * SB, const float * MinB, const uint8_t * B);

class Qwen_Linear_with_bias_Int4 
{
    // Matrix3D<uint8_t> weight; // each uint8_t contains two int4 weights
    // Matrix3D<float> fp32_weight; // for debugging,
    // Matrix3D<float> bias;
    // Matrix3D<float> scale, offset;
    // Matrix3D<int8_t> zero_point; // quantization related parameters
    // Matrix3D<int8_t> activation_int8;
    // Matrix3D<float> activation_scale;
    int weight_cols, weight_rows;
    bool has_bias = false;
    std::string profile_name = "Qwen_Linear_with_bias_Int4";
    ModelContext * context_;

    static void null_deleter(int8_t*) {}
   public:
    std::unique_ptr<int8_t, void(*)(int8_t *)> weight_repack_fp16;
    std::unique_ptr<int8_t, void(*)(int8_t *)> weight_repack_fp32;


    Qwen_Linear_with_bias_Int4():weight_repack_fp16(nullptr, null_deleter),weight_repack_fp32(nullptr, null_deleter)
    {
        weight_cols = 0;
        weight_rows = 0;
        has_bias = false;
    };
    Qwen_Linear_with_bias_Int4(ModelContext * ctx, std::string path, int weight_dim_x, int weight_dim_y,int weight_dim_z):weight_repack_fp16(nullptr, null_deleter),weight_repack_fp32(nullptr, null_deleter)
    {
        IF_DEBUG( printf("Constructor Qwen_Linear_with_bias_Int4 ... \n");)
        context_ = ctx;
        long long weight_size = (long long )weight_dim_x * (long long )weight_dim_y * (long long )weight_dim_z; // total number of weights
        int num_blocks  = weight_size/ QK;

        int8_t * weights_buffer = ctx ->repack_buffer.get();

        // Matrix3D<uint8_t> weight = Matrix3D<uint8_t>(weight_dim_x, weight_dim_y, weight_dim_z / 2);
        // assert (weight_dim_x * weight_dim_y * weight_dim_z / QK  == num_blocks);
        // Matrix3D<float> scale = Matrix3D<float>(weight_dim_x, weight_dim_y, weight_dim_z / QK);
        // Matrix3D<float> offset = Matrix3D<float>(1, 1, 1);
        // Matrix3D<int8_t> zero_point = Matrix3D<int8_t>(weight_dim_x, weight_dim_y, weight_dim_z / QK);

        // Create views into the buffer
        MatrixView<uint8_t> q4_w(reinterpret_cast<uint8_t*>(weights_buffer), weight_dim_x, weight_dim_y, weight_dim_z / 2);
        float * scale_buffer = reinterpret_cast<float*>(weights_buffer + q4_w.length());
        MatrixView<float> scale(scale_buffer, weight_dim_x, weight_dim_y, weight_dim_z / QK);
        float * offset_buffer = scale_buffer + scale.length();
        MatrixView<float> offset(offset_buffer,weight_dim_x, weight_dim_y, weight_dim_z / QK);
        int8_t * zp_buffer = reinterpret_cast<int8_t *>(offset_buffer + offset.length());
        MatrixView<int8_t> zero_point(zp_buffer, weight_dim_x, weight_dim_y, weight_dim_z / QK);

        // Load data directly from disk into buffer
        q4_w.load((path + "weight_int4.bin").c_str());
        scale.load((path + "scaling_factor_int4.bin").c_str());
        offset.load((path + "offset_int4.bin").c_str());
        zero_point.load((path + "zero_point_int4.bin").c_str());

        int num_repack_blocks = weight_dim_y * weight_dim_z / (QK * 2 * 8);
        weight_rows = weight_dim_z;
        weight_cols = weight_dim_y;
        // weight_repack_fp16 = std::make_unique<int8_t[]>( sizeof(q4_repack_2x8_fp16) * num_repack_blocks);
        // weight_repack_fp32 = std::make_unique<int8_t[]>( sizeof(q4_repack_2x8_fp32) * num_repack_blocks);
        weight_repack_fp16 = make_aligned_x86<int8_t>(ALIGNMENT,  sizeof(q4_repack_2x8_fp16) * num_repack_blocks);
        weight_repack_fp32 = make_aligned_x86<int8_t>(ALIGNMENT,  sizeof(q4_repack_2x8_fp32) * num_repack_blocks);

        // repack
        IF_DEBUG(printf("Repacking weight ... \n");)
        // repack_w80_weight(weight_dim_z, weight_dim_y, QK, weight_repack.get(), scale.data(), weight.data());
        repack_w81_weight_fp16(weight_dim_z, weight_dim_y, QK, weight_repack_fp16.get(), scale.data(), offset.data(),q4_w.data());
        repack_w81_weight_fp32(weight_dim_z, weight_dim_y, QK, weight_repack_fp32.get(), scale.data(), offset.data(),q4_w.data());
        IF_DEBUG(printf("weight repacked... \n");)
        
    };
    
    
    Qwen_Linear_with_bias_Int4(const Qwen_Linear_with_bias_Int4 &other)
        : weight_repack_fp16(nullptr, null_deleter),
          weight_repack_fp32(nullptr, null_deleter)
    {
        IF_DEBUG(printf("Copy Constructor Qwen_Linear_with_bias_Int4 ..."));
        weight_rows = other.weight_rows;
        weight_cols = other.weight_cols;
        has_bias = other.has_bias;
        context_ = other.context_;
        // Use the same allocation logic as the main constructor
        // num_repack_blocks = weight_dim_y * weight_dim_z / (QK * 2 * 8);
        int num_repack_blocks = weight_cols * weight_rows / (QK * 2 * 8);
        size_t fp16_size = sizeof(q4_repack_2x8_fp16) * num_repack_blocks;
        size_t fp32_size = sizeof(q4_repack_2x8_fp32) * num_repack_blocks;
        weight_repack_fp16 = make_aligned_x86<int8_t>(ALIGNMENT, fp16_size);
        weight_repack_fp32 = make_aligned_x86<int8_t>(ALIGNMENT, fp32_size);
        std::copy(other.weight_repack_fp16.get(), other.weight_repack_fp16.get() + fp16_size, weight_repack_fp16.get());
        std::copy(other.weight_repack_fp32.get(), other.weight_repack_fp32.get() + fp32_size, weight_repack_fp32.get());
        IF_DEBUG(printf(" Done!\n"));
    }
    
    Qwen_Linear_with_bias_Int4(Qwen_Linear_with_bias_Int4 &&other)
        : weight_repack_fp16(std::move(other.weight_repack_fp16)),
          weight_repack_fp32(std::move(other.weight_repack_fp32))
    {
        IF_DEBUG(printf("Move Constructor Qwen_Linear_with_bias_Int4 ..."));
        weight_rows = other.weight_rows;
        weight_cols = other.weight_cols;
        has_bias = other.has_bias;
        context_ = other.context_;
        IF_DEBUG(printf(" Done!\n"));
    }
    
    Qwen_Linear_with_bias_Int4& operator=(const Qwen_Linear_with_bias_Int4 &other) {
        IF_DEBUG(printf("Copy Assignment Qwen_Linear_with_bias_Int4 ..."));
        if (this == &other) return *this;
        weight_rows = other.weight_rows;
        weight_cols = other.weight_cols;
        has_bias = other.has_bias;
        context_ = other.context_;
        int num_repack_blocks = weight_cols * weight_rows / (QK * 2 * 8);
        size_t fp16_size = sizeof(q4_repack_2x8_fp16) * num_repack_blocks;
        size_t fp32_size = sizeof(q4_repack_2x8_fp32) * num_repack_blocks;
        weight_repack_fp16 = make_aligned_x86<int8_t>(ALIGNMENT, fp16_size);
        weight_repack_fp32 = make_aligned_x86<int8_t>(ALIGNMENT, fp32_size);
        std::copy(other.weight_repack_fp16.get(), other.weight_repack_fp16.get() + fp16_size, weight_repack_fp16.get());
        std::copy(other.weight_repack_fp32.get(), other.weight_repack_fp32.get() + fp32_size, weight_repack_fp32.get());
        IF_DEBUG(printf(" Done!\n"));
        return *this;
    }
    
    Qwen_Linear_with_bias_Int4& operator=(Qwen_Linear_with_bias_Int4 &&other) {
        IF_DEBUG(printf("Move Assignment Qwen_Linear_with_bias_Int4 ..."));
        if (this == &other) return *this;
        weight_rows = other.weight_rows;
        weight_cols = other.weight_cols;
        has_bias = other.has_bias;
        context_ = other.context_;
        weight_repack_fp16 = std::move(other.weight_repack_fp16);
        weight_repack_fp32 = std::move(other.weight_repack_fp32);
        IF_DEBUG(printf(" Done!\n"));
        return *this;
    }

    Matrix3D<float> forward( const Matrix3D<float> &activation);
    Matrix3D<float> forward_debug(const Matrix3D<float> &activation, int8_t * A_repack, float* weight_repack);
    // Matrix3D<float> forward_gemm ( const Matrix3D<float> &activation);
    // method to evaluate the correctness optimization method
    void forward_reference(const Matrix3D<float> &x, Matrix3D<float> &output);
    // void initialize_memory(const int block_size);
    // bool check_weight_quantization_error();
};

class Qwen3RMSNorm{
   public:
    Qwen3RMSNorm(int hidden_dim) {
        weight = Matrix3D<float>(1, 1, hidden_dim);
    };
    Qwen3RMSNorm(){};
    void load(std::string path);
    Matrix3D<float> forward(const Matrix3D<float> &x,int dim = -1);
    Matrix3D<float> weight;
    float eps = 1e-6;

   private:
    std::string profile_name = "Qwen3RMSNorm";

};

// batch GEMX implementation of Group Query Attention
class bgemxGQA {
   public:
    bgemxGQA(float _alpha, int num_q_head, int num_kv_head);
    bgemxGQA(){};
    void forward_openblas_pv(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    void forward_openblas_qk(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    void forward_ink_kernel_qk(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    void forward_ink_kernel_pv(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    void forward_mix_kernel_pv(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    void forward(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    void forward_weight_untransposed(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    float alpha;
    int groupsize;

   private:
    std::string profile_name = "bgemmGQA";
};

void load_BMM_F32T(bgemxGQA &op, std::string prefix);


Matrix3D<float> Qwen3SiLuMul(const Matrix3D<float> &a, const Matrix3D<float> &b) ;

Matrix3D<float> add(Matrix3D<float> &a, Matrix3D<float> &b) ;


void quantize_row_q8_0_repack(const float * x, void * vy, int64_t k);
void quantize_row_q8_1_repack_fp16(const float * x, void * vy, int64_t k);
void quantize_row_q8_1_repack_fp32(const float * x, void * vy, int64_t k);

void gemm_repack_A80W40(
    void * A_repack,
    void * B_repack,
    float* C,
    const int M, const int N, const int K
);
void gemv_repack_A80W40(
    void * A_repack,
    void * B_repack,
    float* C,
    const int M, const int N, const int K
);

void gemm_repack_A81W41_fp32(
    void * A_repack,
    void * B_repack,
    float* C,
    const int M, const int N, const int K
);
void gemv_repack_A81W41_fp16(
    void * A_repack,
    void * B_repack,
    float* C,
    const int M, const int N, const int K
);
void gemm_fp32_rcr(float* A, float* B, float* C, const int M, const int N, const int K);
void gemv_fp32_rcr_mt_impl_1(float* A, float* B, float* C, const int M, const int N, const int K); 

void batch_gemv_fp32_rcr_naive(const int n_heads, float** A_pointers, float** B_pointers, float** C_pointers, const int M, const int N, const int K);
void batch_gemv_fp32_rcr_avx(const int n_heads, float** A_pointers, float** B_pointers, float** C_pointers, const int M, const int N, const int K);

void batch_gemv_fp32_rrr_naive(const int n_heads, float** A_pointers, float** B_pointers, float** C_pointers, const int M, const int N, const int K);
void batch_gemv_fp32_rrr_avx(const int n_heads, float** A_pointers, float** B_pointers, float** C_pointers, const int M, const int N, const int K);

void gemv_fp32_rcr_mt_impl_2(float* A, float* B, float* C, const int M, const int N, const int K);
void gemm_fp32_rrr(float* A, float* B, float* C, const int M, const int N, const int K);
void gemv_fp32_rrr(float* A, float* B, float* C, const int M, const int N, const int K);
void gemv_fp32_rrr_naive(float* A, float* B, float* C, const int M, const int N, const int K);
void gemv_fp32_rrr_dummy(float* A, float* B, float* C, const int M, const int N, const int K);

bool has_nan(Matrix3D<float> &mat);
#endif