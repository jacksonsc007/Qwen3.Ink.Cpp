#include <cstring>

#include "common.h"
#include "operators.h"
#include "utils.h"

void Embedding::load(std::string prefix) {
    lookup.load((prefix + "/weight.bin").c_str());
}

Matrix3D<float> Embedding::forward(Matrix3D<int> input_id) {
    PROFILE_START(profile_name);
    assert(input_id.m_dim_x == 1);
    int bs = (input_id.m_dim_y);
    int sqlen = (input_id.m_dim_z);
    Matrix3D<float> output(bs, sqlen, embed_dim);
    
    for (int i = 0; i < input_id.m_dim_z; i++) {
        int token_id = input_id(0, 0, i);
        float* output_sample_ptr = &output.data()[i * this->embed_dim];
        float* target_embed = &this->lookup.data()[token_id * this->embed_dim];
        memcpy(output_sample_ptr, target_embed, sizeof(float) * this->embed_dim);
    }
    PROFILE_END(profile_name);
    return output;
}
