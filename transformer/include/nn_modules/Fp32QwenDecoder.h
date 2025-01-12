#ifndef _Fp32QwenDecoder_h
#define _Fp32QwenDecoder_h

#include <vector>

#include "Fp32QwenDecoderLayer.h"
#include "common.h"
#include "operators.h"
#include "Fp32QwenOperator.h"

struct Fp32QwenModel_output {
    Matrix3D<float> last_hidden_state;
    std::vector<Matrix3D<float>> past_keys, past_values;
};

struct Fp32QwenModel_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Fp32QwenModel_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Fp32QwenModel_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                           std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};


class Fp32QwenModel {
public:
    // member
    int voc_size, embed_dim, padding_idx, hidden_dim, num_heads;
    Embedding wte;
    QwenRMSNorm ln_F;
    std::vector<Fp32QwenBlock> layers;
    std::string profile_name = "Fp32QwenModel";
    float* attention_mask_buf;
    float* last_hidden_states_buf;

    // method
    Fp32QwenModel(){};
    Fp32QwenModel(std::string param_path, const struct qwen_config config);
    Fp32QwenModel_output forward(const struct Fp32QwenModel_input &input);
    Matrix3D<float> prepare_decoder_attention_mask(int length, int past_length);



};

#endif