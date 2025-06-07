#ifndef _QwenDecoderLayer_H
#define _QwenDecoderLayer_H

#include "QwenOperator.h"
#include "QwenAttention.h"

struct Qwen3DecoderLayer_Input{
    Matrix3D<float> hidden_states_arr;
    Matrix3D<float> attention_mask;
    int past_sqlen;

    Qwen3DecoderLayer_Input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_, int past_sqlen_) {
        hidden_states_arr = hidden_states_;
        attention_mask = attention_mask_;
        past_sqlen = past_sqlen_;
    }

};
struct Qwen3DecoderLayer_Output{
    Matrix3D<float> hidden_states;
    Matrix3D<float> attentions;

    Qwen3DecoderLayer_Output(Matrix3D<float> hidden_states_, Matrix3D<float> attentions_) {
        hidden_states = hidden_states_;
        attentions = attentions_;
    };
};

class Qwen3DecoderLayer{
    ModelContext * context_ptr;
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
    Qwen3DecoderLayer(ModelContext * ctx, std::string param_path, const struct qwen3_config config, int layer_idx);
    Qwen3DecoderLayer_Output forward(const Qwen3DecoderLayer_Input &input);


};

#endif