#ifndef _Int4QwenDecoderLayer_H
#define _Int4QwenDecoderLayer_H

#include "Int4QwenOperator.h"
#include "Int4QwenAttention.h"

struct Int4QwenBlock_input{
    Matrix3D<float> hidden_states_arr;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key;
    Matrix3D<float> past_value;
    bool has_past_key_value = false;
    Int4QwenBlock_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_) {
        hidden_states_arr = hidden_states_;
        attention_mask = attention_mask_;
        has_past_key_value = false;
    }

    Int4QwenBlock_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_,
                                Matrix3D<float> past_key_, Matrix3D<float> past_value_) {
        hidden_states_arr = hidden_states_;
        attention_mask = attention_mask_;
        past_key = past_key_;
        past_value = past_value_;
        has_past_key_value = true;
    }

};
struct Int4QwenBlock_output{
    Matrix3D<float> hidden_states;
    Matrix3D<float> attentions;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;

    Int4QwenBlock_output(Matrix3D<float> hidden_states_, Matrix3D<float> attentions_,
                                 std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value_) {
        hidden_states = hidden_states_;
        attentions = attentions_;
        past_key_value = past_key_value_;
    };
};

class Int4QwenBlock{
public:
    // member
    int embed_dim, num_attention_heads, hidden_dim, layer_idx;
    Int4QwenAttention attn;
    QwenInt4RMSNorm ln_1;
    QwenInt4RMSNorm ln_2;
    // QwenMLP mlp;
    Qwen_Linear_with_bias_Int4 gate_proj, down_proj, up_proj;

    std::string profile_name = "Int4QwenBlock";
    
    // method
    Int4QwenBlock() = default;
    Int4QwenBlock(std::string param_path, const struct qwen_config config, int layer_idx);
    Int4QwenBlock_output forward(const Int4QwenBlock_input &input);


};

#endif