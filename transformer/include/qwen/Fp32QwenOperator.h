#ifndef _FP32QWENOPERATOR_H
#define _FP32QWENOPERATOR_H

#include "common.h"
#include "utils.h"

class Qwen_Linear_FP {
   public:
    Qwen_Linear_FP(Matrix3D<float> weight_, std::string weight_path, Matrix3D<float> bias_, std::string bias_path) : 
    weight(weight_), bias(bias_) {
        read_to_array((weight_path).c_str(), this->weight.m_data, this->weight.length());
        read_to_array((bias_path).c_str(), this->bias.m_data, this->bias.length());
    };
    Qwen_Linear_FP(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    Matrix3D<float> weight;
    Matrix3D<float> bias;

   private:
    std::string profile_name = "Qwen_Linear_FP";
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