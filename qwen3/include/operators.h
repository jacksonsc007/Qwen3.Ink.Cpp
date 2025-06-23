#ifndef OPERATORS_H
#define OPERATORS_H
#include <cassert>

#include "common.h"
#include "ops/Embedding.h"
#include "ops/RotaryPosEmb.h"

void softmax(const Matrix3D<float> &input, Matrix3D<float> &output);
void batch_Add(const Matrix3D<float> &input, const Matrix3D<float> &input2, Matrix3D<float> &output);
template <typename T>
void linear(Matrix3D<T> &a, Matrix3D<T> &b, Matrix3D<T> &c);
#endif  // OPERATORS_H
