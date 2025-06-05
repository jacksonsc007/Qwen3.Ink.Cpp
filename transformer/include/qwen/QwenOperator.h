#ifndef _QWENOPERATOR_H
#define _QWENOPERATOR_H

#include "common.h"
#include "utils.h"

bool has_nan(Matrix3D<float> mat);
void permute01(Matrix3D<float> before, Matrix3D<float> after);
void reshape_headfirst(Matrix3D<float> before, Matrix3D<float> after);
void reshape_seqfirst(Matrix3D<float> before, Matrix3D<float> after);


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
        allocate_aligned_memory(weight_arr,  weight_size * sizeof(uint8_t) / 2); // each uint8_t contains two int4 weights
        allocate_aligned_memory(scale_arr,  num_blocks * sizeof(float));
        allocate_aligned_memory(offset_arr,  1 * sizeof(float));
        allocate_aligned_memory(zero_point_arr,  num_blocks * sizeof(int8_t));

        weight = Matrix3D<uint8_t>(weight_arr, weight_dim_x, weight_dim_y, weight_dim_z / 2);
        assert (weight_dim_x * weight_dim_y * weight_dim_z / QK  == num_blocks);
        scale = Matrix3D<float>(scale_arr, weight_dim_x, weight_dim_y, weight_dim_z / QK);
        offset = Matrix3D<float>(offset_arr, 1, 1, 1);
        zero_point = Matrix3D<int8_t>(zero_point_arr, weight_dim_x, weight_dim_y, weight_dim_z / QK);
        
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
        allocate_aligned_memory(weight_arr,  weight_size * sizeof(uint8_t) / 2); // each uint8_t contains two int4 weights
        // TODOink: quantize bias?
        allocate_aligned_memory(bias_arr,  bias_size * sizeof(float));
        allocate_aligned_memory(scale_arr,  num_blocks * sizeof(float));
        allocate_aligned_memory(offset_arr,  1 * sizeof(float));
        allocate_aligned_memory(zero_point_arr,  num_blocks * sizeof(int8_t));

        weight = Matrix3D<uint8_t>(weight_arr, weight_dim_x, weight_dim_y, weight_dim_z / 2);
        assert (weight_dim_x * weight_dim_y * weight_dim_z / QK  == num_blocks);
        bias = Matrix3D<float>(bias_arr, bias_dim_x, bias_dim_y, bias_dim_z);
        scale = Matrix3D<float>(scale_arr, weight_dim_x, weight_dim_y, weight_dim_z / QK);
        offset = Matrix3D<float>(offset_arr, 1, 1, 1);
        zero_point = Matrix3D<int8_t>(zero_point_arr, weight_dim_x, weight_dim_y, weight_dim_z / QK);
        
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
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    // method to evaluate the correctness optimization method
    void forward_reference(const Matrix3D<float> &x, Matrix3D<float> &output);
    void initialize_memory(const int block_size);
    // bool check_weight_quantization_error();
    Matrix3D<uint8_t> weight; // each uint8_t contains two int4 weights
    Matrix3D<float> fp32_weight; // for debugging,
    Matrix3D<float> bias;
    Matrix3D<float> scale, offset;
    Matrix3D<int8_t> zero_point; // quantization related parameters
    bool has_bias = false;

    std::string profile_name = "Qwen_Linear_with_bias_Int4";
};

class Qwen3RMSNorm{
   public:
    Qwen3RMSNorm(Matrix3D<float> _weight) : weight(_weight){};
    Qwen3RMSNorm(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output, int dim = -1);
    Matrix3D<float> weight;
    float eps = 1e-6;

   private:
    std::string profile_name = "Qwen3RMSNorm";

};


#endif