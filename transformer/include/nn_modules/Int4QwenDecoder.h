#ifndef _Int4QwenDecoder_h
#define _Int4QwenDecoder_h

#include <vector>

#include "Int4QwenDecoderLayer.h"
#include "common.h"
#include "operators.h"
#include "Int4QwenOperator.h"

struct Int4QwenModel_output {
    Matrix3D<float> last_hidden_state;
    std::vector<Matrix3D<float>> past_keys, past_values;
};

struct Int4QwenModel_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Int4QwenModel_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Int4QwenModel_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                           std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};


class Int4QwenModel {
public:
    // member
    int voc_size, embed_dim, padding_idx, hidden_dim, num_heads;
    Embedding wte;
    QwenInt4RMSNorm ln_F;
    std::vector<Int4QwenBlock> layers;
    std::string profile_name = "Int4QwenModel";
    float* attention_mask_buf;
    float* last_hidden_states_buf;

    // method
    Int4QwenModel(){};
    Int4QwenModel(std::string param_path, const struct qwen_config config);
    Int4QwenModel_output forward(const struct Int4QwenModel_input &input);
    Matrix3D<float> prepare_decoder_attention_mask(int length, int past_length);



};

#endif