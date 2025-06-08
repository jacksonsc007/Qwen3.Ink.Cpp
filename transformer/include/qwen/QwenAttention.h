#pragma once
#include "common.h"
#include "operators.h"
#include "QwenOperator.h"


struct Qwen3Attention_Output {
    Matrix3D<float> attn_output;
    Matrix3D<float> attn_probs_reshaped;
};

struct Qwen3Attention_Input
{
    Matrix3D<float> hidden_state; //(bs, sqlen, embed_dim)
    Matrix3D<float> attention_mask;
    int layer_idx;
    int past_sqlen;
    
    Qwen3Attention_Input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, int layer_idx_, int past_sqlen_)
        : hidden_state(hidden_states_), attention_mask(attention_mask_), layer_idx(layer_idx_), past_sqlen(past_sqlen_) {}

};


class Qwen3Attention {
    float * k_cache_space, * v_cache_space;

public:
    // construct
    Qwen3Attention(){}
    Qwen3Attention(float * k_cache_space_, float * v_cache_space_, std::string param_path, struct qwen3_config config, int layer_idx);

    // member
    int hidden_dim, head_dim, max_sqlen;
    int layer_idx;
    std::string params_path;
    int num_q_head, num_kv_head;
    int q_dim, kv_dim;
    Qwen_Linear_with_bias_Int4 k_proj, v_proj, q_proj, o_proj;
    Qwen3RMSNorm q_norm, k_norm;
    RotaryPosEmb rope_embed;
    bgemmGQA qk_bmm, pv_bmm;
    std::string profile_name = "Qwen3Attention";
    std::string forward_profile_name;

    // method
    struct Qwen3Attention_Output forward(struct Qwen3Attention_Input &input);
};