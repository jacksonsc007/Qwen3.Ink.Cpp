#ifndef _QWENOPERATOR_H
#define _QWENOPERATOR_H

#include <cstddef>
#include <cstdint>
#include "common.h"
#include "utils.h"
#include <cassert>
#include <stdexcept>

bool has_nan(Matrix3D<float> mat);
void permute01(Matrix3D<float> before, Matrix3D<float> after);
void reshape_headfirst(Matrix3D<float> before, Matrix3D<float> after);
void reshape_seqfirst(Matrix3D<float> before, Matrix3D<float> after);




struct ModelContext{
    std::unique_ptr<float[]> k_cache;
    std::unique_ptr<float[]> v_cache;
};



template <typename T>
class MatrixView {
public:
    T* m_data;
    int m_dim_x, m_dim_y, m_dim_z;          // Current dimensions (including repeats)
    int m_dim_x_original, m_dim_y_original, m_dim_z_original; // Original dimensions
    int stride_x, stride_y, stride_z;
    int m_repeat_x, m_repeat_y, m_repeat_z; // Repeat counts per dimension
    MatrixView()
    {
        m_data = NULL;
        m_dim_x = 0;
        m_dim_y = 0;
        m_dim_z = 0;
        m_dim_x_original = 0;
        m_dim_y_original = 0;
        m_dim_z_original = 0;
        stride_x = 0;
        stride_y = 0;
        stride_z = 0;
        m_repeat_x = 1;
        m_repeat_y = 1;
        m_repeat_z = 1;
    }
    // Constructor - wraps existing memory with optional strides
    MatrixView(T* data, int dim_x, int dim_y, int dim_z,
               int stride_x = 0, int stride_y = 0, int stride_z = 0)
        : m_data(data),
          m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z),
          m_dim_x_original(dim_x), m_dim_y_original(dim_y), m_dim_z_original(dim_z),
          stride_x(stride_x > 0 ? stride_x : dim_y * dim_z),
          stride_y(stride_y > 0 ? stride_y : dim_z),
          stride_z(stride_z > 0 ? stride_z : 1),
          m_repeat_x(1), m_repeat_y(1), m_repeat_z(1)
    {
        if (!data) {
            throw std::invalid_argument("Data pointer cannot be null");
        }
        if (dim_x <= 0 || dim_y <= 0 || dim_z <= 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
    }
    
    MatrixView(const MatrixView<T> &other)
    {
        m_data = other.m_data;
        m_dim_x = other.m_dim_x;
        m_dim_y = other.m_dim_y;
        m_dim_z = other.m_dim_z;
        m_dim_x_original = other.m_dim_x_original;
        m_dim_y_original = other.m_dim_y_original;
        m_dim_z_original = other.m_dim_z_original;
        stride_x = other.stride_x;
        stride_y = other.stride_y;
        stride_z = other.stride_z;
        m_repeat_x = other.m_repeat_x;
        m_repeat_y = other.m_repeat_y;
        m_repeat_z = other.m_repeat_z;
    }
    
    Matrix3D<T> contiguous()
    {
        Matrix3D<T> output(m_dim_x, m_dim_y, m_dim_z);
        for (int i = 0; i < m_dim_x; i++)
        {
            for (int j = 0; j < m_dim_y; j++)
            {
                for (int k = 0; k < m_dim_z; k++)
                {
                    output(i, j, k) = (*this)(i, j, k);
                }
            }
        }
        return output;
    }

    // Access element with bounds checking and repetition handling
    T& operator()(int x, int y, int z) {
        // modify_repetition_index(x, y, z);
        // check_bounds(x, y, z);
        return m_data[x * stride_x + y * stride_y + z * stride_z];
    }

    const T& operator()(int x, int y, int z) const {
        // modify_repetition_index(x, y, z);
        // check_bounds(x, y, z);
        return m_data[x * stride_x + y * stride_y + z * stride_z];
    }

    // Create a view with a dimension repeated
    MatrixView<T> repeat_dimension(int dim, int times) const {
        if (times <= 0) {
            throw std::invalid_argument("Repeat times must be positive");
        }

        MatrixView<T> new_view = *this;
        switch (dim) {
            case 0: 
                new_view.m_dim_x *= times;
                new_view.m_repeat_x *= times;
                break;
            case 1:
                new_view.m_dim_y *= times;
                new_view.m_repeat_y *= times;
                break;
            case 2:
                new_view.m_dim_z *= times;
                new_view.m_repeat_z *= times;
                break;
            default:
                throw std::invalid_argument("Invalid dimension (0=x, 1=y, 2=z)");
        }
        return new_view;
    }

    // Create subview
    MatrixView<T> subview(int x_start, int x_end,
                          int y_start, int y_end,
                          int z_start, int z_end) const {
        return MatrixView<T>(
            &(*this)(x_start, y_start, z_start),
            x_end - x_start,
            y_end - y_start,
            z_end - z_start,
            stride_x,
            stride_y,
            stride_z
        );
    }

    // Raw data access
    T* data() { return m_data; }
    const T* data() const { return m_data; }

    // Dimensions
    int dim_x() const { return m_dim_x; }
    int dim_y() const { return m_dim_y; }
    int dim_z() const { return m_dim_z; }

    // Original dimensions (without repetition)
    int original_dim_x() const { return m_dim_x_original; }
    int original_dim_y() const { return m_dim_y_original; }
    int original_dim_z() const { return m_dim_z_original; }


    // Repeat counts
    int repeat_x() const { return m_repeat_x; }
    int repeat_y() const { return m_repeat_y; }
    int repeat_z() const { return m_repeat_z; }

    void modify_repetition_index(int& x, int& y, int& z) const {
        if (m_repeat_x > 1) x = x / m_repeat_x;
        if (m_repeat_y > 1) y = y / m_repeat_y;
        if (m_repeat_z > 1) z = z / m_repeat_z;
    }

    void check_bounds(int x, int y, int z) const {
        if (x < 0 || x >= m_dim_x_original ||
            y < 0 || y >= m_dim_y_original ||
            z < 0 || z >= m_dim_z_original) {
            throw std::out_of_range(
                "MatrixView index (" + std::to_string(x) + "," 
                + std::to_string(y) + "," + std::to_string(z) 
                + ") out of bounds for original dimensions ("
                + std::to_string(m_dim_x_original) + ","
                + std::to_string(m_dim_y_original) + ","
                + std::to_string(m_dim_z_original) + ")");
        }
    }
};

class Qwen_Linear_with_bias_Int4 
{
   public:
    /**
     * @brief 构造函数，用于初始化 Qwen_Linear_Int4 对象。
     *
     * @param 
     * @param 
     * @param 
     */
    Qwen_Linear_with_bias_Int4(std::string path, int weight_dim_x, int weight_dim_y,int weight_dim_z) 
    {
        uint8_t * weight_arr;
        float * scale_arr,  * offset_arr;
        int8_t * zero_point_arr;
        long long weight_size = (long long )weight_dim_x * (long long )weight_dim_y * (long long )weight_dim_z; // total number of weights
        int num_blocks  = weight_size/ QK;

        weight = Matrix3D<uint8_t>(weight_dim_x, weight_dim_y, weight_dim_z / 2);
        assert (weight_dim_x * weight_dim_y * weight_dim_z / QK  == num_blocks);
        scale = Matrix3D<float>(weight_dim_x, weight_dim_y, weight_dim_z / QK);
        offset = Matrix3D<float>(1, 1, 1);
        zero_point = Matrix3D<int8_t>(weight_dim_x, weight_dim_y, weight_dim_z / QK);
        
        weight.load((path     + "weight_int4.bin").c_str());
        // offset.load((path     + "offset_int4.bin").c_str()); 
        scale.load((path      + "scaling_factor_int4.bin").c_str());
        zero_point.load((path + "zero_point_int4.bin").c_str());
        has_bias = false;

        // debugging
        #ifdef qwen_debug_fp32
        float * fp32_weight_arr;
        allocate_aligned_memory(fp32_weight_arr,  weight_size * sizeof(float));
        fp32_weight = Matrix3D<float>(fp32_weight_arr, weight_dim_x, weight_dim_y, weight_dim_z);
        fp32_weight.load((path     + "weight_fp32.bin").c_str());
        #endif
    };

    Qwen_Linear_with_bias_Int4(std::string path, int weight_dim_x, int weight_dim_y,int weight_dim_z, 
                                       int bias_dim_x, int bias_dim_y,int bias_dim_z) 
    {
        uint8_t * weight_arr;
        float * bias_arr;
        float * scale_arr,  * offset_arr;
        int8_t * zero_point_arr;
        long long weight_size = (long long )weight_dim_x * (long long )weight_dim_y * (long long )weight_dim_z; // total number of weights
        long long bias_size = bias_dim_x * bias_dim_y * bias_dim_z; // total number of bias
        int num_blocks  = weight_size/ QK;

        weight = Matrix3D<uint8_t>(weight_dim_x, weight_dim_y, weight_dim_z / 2);
        assert (weight_dim_x * weight_dim_y * weight_dim_z / QK  == num_blocks);
        bias = Matrix3D<float>(bias_dim_x, bias_dim_y, bias_dim_z);
        scale = Matrix3D<float>(weight_dim_x, weight_dim_y, weight_dim_z / QK);
        offset = Matrix3D<float>(1, 1, 1);
        zero_point = Matrix3D<int8_t>(weight_dim_x, weight_dim_y, weight_dim_z / QK);
        
        weight.load((path     + "weight_int4.bin").c_str());
        bias.load((path       + "bias.bin").c_str()); // TODO: could bias be quantized?
        // offset.load((path     + "offset_int4.bin").c_str()); 
        scale.load((path      + "scaling_factor_int4.bin").c_str());
        zero_point.load((path + "zero_point_int4.bin").c_str());
        has_bias = true;
        
        #ifdef qwen_debug_fp32
        float * fp32_weight_arr;
        allocate_aligned_memory(fp32_weight_arr,  weight_size * sizeof(float));
        fp32_weight = Matrix3D<float>(fp32_weight_arr, weight_dim_x, weight_dim_y, weight_dim_z);
        fp32_weight.load((path     + "weight_fp32.bin").c_str());
        #endif


    };
    Qwen_Linear_with_bias_Int4(){};
    Matrix3D<float> forward( Matrix3D<float> &activation);
    // method to evaluate the correctness optimization method
    void forward_reference(const Matrix3D<float> &x, Matrix3D<float> &output);
    void initialize_memory(const int block_size);
    // bool check_weight_quantization_error();
    Matrix3D<uint8_t> weight; // each uint8_t contains two int4 weights
    Matrix3D<float> fp32_weight; // for debugging,
    Matrix3D<float> bias;
    Matrix3D<float> scale, offset;
    Matrix3D<int8_t> zero_point; // quantization related parameters
    Matrix3D<int8_t> activation_int8;
    Matrix3D<float> activation_scale;
    bool has_bias = false;

    std::string profile_name = "Qwen_Linear_with_bias_Int4";
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

class bgemmGQA {
   public:
    bgemmGQA(float _alpha, int num_q_head, int num_kv_head);
    bgemmGQA(){};
    void forward_openblas_pv(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    void forward_openblas_qk(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    void forward(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    void forward_weight_untransposed(Matrix3D<float> &A, MatrixView<float> &B, Matrix3D<float> &output);
    float alpha;
    int groupsize;

   private:
    std::string profile_name = "bgemmGQA";
};

void load_BMM_F32T(bgemmGQA &op, std::string prefix);

#endif