#ifndef _QwenDecoderLayer_H
#define _QwenDecoderLayer_H

#include "QwenOperator.h"
#include "QwenAttention.h"

struct Qwen3DecoderLayer_Input{
    Matrix3D<float> hidden_states_arr;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key;
    Matrix3D<float> past_value;
    bool has_past_key_value = false;
    Qwen3DecoderLayer_Input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_) {
        hidden_states_arr = hidden_states_;
        attention_mask = attention_mask_;
        has_past_key_value = false;
    }

    Qwen3DecoderLayer_Input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_,
                                Matrix3D<float> past_key_, Matrix3D<float> past_value_) {
        hidden_states_arr = hidden_states_;
        attention_mask = attention_mask_;
        past_key = past_key_;
        past_value = past_value_;
        has_past_key_value = true;
    }

};
struct Qwen3DecoderLayer_Output{
    Matrix3D<float> hidden_states;
    Matrix3D<float> attentions;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;

    Qwen3DecoderLayer_Output(Matrix3D<float> hidden_states_, Matrix3D<float> attentions_,
                                 std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value_) {
        hidden_states = hidden_states_;
        attentions = attentions_;
        past_key_value = past_key_value_;
    };
};

class Qwen3DecoderLayer{
public:
    // member
    int hidden_dim, max_sqlen, num_heads , layer_idx;
    int mlp_proj_dim = 12288;
    Qwen3Attention attn;
    Qwen3RMSNorm input_layernorm;
    Qwen3RMSNorm post_attention_layernorm;
    // QwenMLP mlp;
    Qwen_Linear_with_bias_Int4 gate_proj, down_proj, up_proj;

    std::string profile_name = "Qwen3DecoderLayer";
    std::string forward_profile_name;
    
    // method
    Qwen3DecoderLayer() = default;
    Qwen3DecoderLayer(std::string param_path, const struct qwen3_config config, int layer_idx);
    Qwen3DecoderLayer_Output forward(const Qwen3DecoderLayer_Input &input);


};

#endif