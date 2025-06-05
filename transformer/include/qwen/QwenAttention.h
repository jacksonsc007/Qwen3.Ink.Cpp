#pragma once
#include "common.h"
#include "operators.h"
#include "QwenOperator.h"


struct Qwen3Attention_Output {
    Matrix3D<float> attn_output;
    Matrix3D<float> attn_probs_reshaped;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;
};

struct Qwen3Attention_Input
{
    Matrix3D<float> hidden_state; //(bs, sqlen, embed_dim)
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key;
    Matrix3D<float> past_value;
    bool has_past_key_value = false;
    int layer_idx;
    
    Qwen3Attention_Input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, int layer_idx_)
        : hidden_state(hidden_states_), attention_mask(attention_mask_), layer_idx(layer_idx_) {}

    Qwen3Attention_Input(
        Matrix3D<float> hidden_state_, Matrix3D<float> attention_mask_, Matrix3D<float> past_key_, Matrix3D<float> past_value_, 
        bool has_past_key_value_, int layer_idx_
    ): 
        hidden_state(hidden_state_), 
        attention_mask(attention_mask_), 
        past_key(past_key_), 
        past_value(past_value_),
        has_past_key_value(has_past_key_value_),
        layer_idx(layer_idx_){}
};


class Qwen3Attention {
public:
    // construct
    Qwen3Attention(){}
    Qwen3Attention(std::string param_path, const struct qwen3_config config);

    // member
    int hidden_dim, head_dim, max_sqlen;
    int num_q_head, num_kv_head;
    int q_dim, kv_dim;
    Qwen_Linear_with_bias_Int4 k_proj, v_proj, q_proj, o_proj;
    Qwen3RMSNorm q_norm, k_norm;
    RotaryPosEmb rope_embed;
    BMM_F32T qk_bmm, pv_bmm;
    std::string profile_name = "Qwen3Attention";

    // method
    static void initialize_memory(const struct qwen3_config config);
    struct Qwen3Attention_Output forward(const struct Qwen3Attention_Input &input);
};