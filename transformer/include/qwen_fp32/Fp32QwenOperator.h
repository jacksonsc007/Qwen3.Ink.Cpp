#ifndef _FP32QWENOPERATOR_H
#define _FP32QWENOPERATOR_H

#include "common.h"
#include "utils.h"

bool has_nan(Matrix3D<float> mat);
void permute01(Matrix3D<float> before, Matrix3D<float> after);
void reshape_headfirst(Matrix3D<float> before, Matrix3D<float> after);
void reshape_seqfirst(Matrix3D<float> before, Matrix3D<float> after);


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