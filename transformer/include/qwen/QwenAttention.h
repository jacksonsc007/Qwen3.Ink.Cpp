#pragma once
#include "common.h"
#include "model.h"
#include "operators.h"
#include "QwenOperator.h"


struct Qwen3Attention_Output {
    Matrix3D<float> attn_output;

    // Default move support
    Qwen3Attention_Output(Qwen3Attention_Output&&) = default;
    Qwen3Attention_Output& operator=(Qwen3Attention_Output&&) = default;

    // Delete copy operations, as we desire RVO
    Qwen3Attention_Output(const Qwen3Attention_Output&) = delete;
    Qwen3Attention_Output& operator=(const Qwen3Attention_Output&) = delete;

    // Or temporarily enable logging
    // Qwen3Attention_Output(const Qwen3Attention_Output& other) {
    //     std::cerr << "ERROR: Copying Qwen3Attention_Output!" << std::endl;
    //     throw std::runtime_error("Copy not allowed");
    // }
    // Qwen3Attention_Output& operator=(const Qwen3Attention_Output&) = delete;

    // Optional: custom constructor
    Qwen3Attention_Output(Matrix3D<float>&& out)
        : attn_output(std::move(out)){}
};


struct Qwen3Attention_Input
{
    const Matrix3D<float>* hidden_state; //(bs, sqlen, embed_dim)
    const Matrix3D<float>* attention_mask;
    int layer_idx;
    int past_sqlen;
    
    Qwen3Attention_Input(const Matrix3D<float> * hidden_states_, const Matrix3D<float>* attention_mask_, int layer_idx_, int past_sqlen_)
        : hidden_state(hidden_states_), attention_mask(attention_mask_), layer_idx(layer_idx_), past_sqlen(past_sqlen_) {}

};


class Qwen3Attention {
    float * k_cache_space, * v_cache_space;
    ModelContext * context_;

public:
    // construct
    Qwen3Attention(){}
    Qwen3Attention(ModelContext * ctx, float * k_cache_space_, float * v_cache_space_, std::string param_path, struct qwen3_config config, int layer_idx);

    // member
    int hidden_dim, head_dim, max_sqlen;
    int layer_idx;
    std::string params_path;
    int num_q_head, num_kv_head;
    int q_dim, kv_dim;
    Qwen_Linear_with_bias_Int4 qkv_proj, o_proj;
    Qwen3RMSNorm q_norm, k_norm;
    RotaryPosEmb rope_embed;
    bgemmGQA qk_bmm, pv_bmm;
    std::string profile_name = "Qwen3Attention";
    std::string forward_profile_name;

    // method
    struct Qwen3Attention_Output forward(struct Qwen3Attention_Input &input);
};