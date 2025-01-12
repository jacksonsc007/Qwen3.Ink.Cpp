#ifndef _Fp32QwenDecoderLayer_H
#define _Fp32QwenDecoderLayer_H

#include "Fp32QwenOperator.h"
#include "Fp32QwenAttention.h"

struct Fp32QwenBlock_input{
    Matrix3D<float> hidden_states_arr;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key;
    Matrix3D<float> past_value;
    bool has_past_key_value = false;
    Fp32QwenBlock_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_) {
        hidden_states_arr = hidden_states_;
        attention_mask = attention_mask_;
        has_past_key_value = false;
    }

    Fp32QwenBlock_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_,
                                Matrix3D<float> past_key_, Matrix3D<float> past_value_) {
        hidden_states_arr = hidden_states_;
        attention_mask = attention_mask_;
        past_key = past_key_;
        past_value = past_value_;
        has_past_key_value = true;
    }

};
struct Fp32QwenBlock_output{
    Matrix3D<float> hidden_states;
    Matrix3D<float> attentions;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;

    Fp32QwenBlock_output(Matrix3D<float> hidden_states_, Matrix3D<float> attentions_,
                                 std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value_) {
        hidden_states = hidden_states_;
        attentions = attentions_;
        past_key_value = past_key_value_;
    };
};

class Fp32QwenBlock{
public:
    // member
    int embed_dim, num_attention_heads, hidden_dim, layer_idx;
    Fp32QwenAttention attn;
    QwenRMSNorm ln_1;
    QwenRMSNorm ln_2;
    // QwenMLP mlp;
    Linear_FP gate_proj, down_proj, up_proj;

    std::string profile_name = "Fp32QwenBlock";
    
    // method
    Fp32QwenBlock() = default;
    Fp32QwenBlock(std::string param_path, const struct qwen_config config, int layer_idx);
    Fp32QwenBlock_output forward(const Fp32QwenBlock_input &input);


};

#endif