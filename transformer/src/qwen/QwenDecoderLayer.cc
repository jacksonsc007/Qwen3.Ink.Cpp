#include "QwenOperator.h"
#include "common.h"
#include "utils.h"
#include "QwenDecoderLayer.h"



Qwen3DecoderLayer::Qwen3DecoderLayer(ModelContext * ctx, std::string param_path, const struct qwen3_config config, int layer_idx)
{
    context_ptr = ctx;
    this->layer_idx = layer_idx;
    max_sqlen = config.max_sqlen;
    hidden_dim = config.hidden_dim;

    // input layernorm
    this->input_layernorm = Qwen3RMSNorm(hidden_dim);
    IF_DEBUG( printf("\e[31m[INFO]\e[m Loading input layernorm for Qwen Block %d...\n", layer_idx););
    input_layernorm.load(param_path + "/input_layernorm/weight.bin");

    IF_DEBUG( printf("\e[31m[INFO]\e[m Loading post attention layernorm for Qwen Block %d...\n", layer_idx););
    // post attention layernorm
    this->post_attention_layernorm = Qwen3RMSNorm(hidden_dim);
    post_attention_layernorm.load(param_path + "/post_attention_layernorm/weight.bin");

    IF_DEBUG( printf("\e[31m[INFO]\e[m Loading self attention layer for Qwen Block %d...\n", layer_idx););
    // attention module
    int layer_stride = config.num_kv_head * max_sqlen * config.head_dim;
    float * k_cache_space = ctx->k_cache.get() + layer_idx * layer_stride;
    float * v_cache_space = ctx->v_cache.get() + layer_idx * layer_stride;
    this -> attn = Qwen3Attention(k_cache_space, v_cache_space, param_path + "/self_attn", config, layer_idx);

    // mlp module
    IF_DEBUG( printf("\e[31m[INFO]\e[m Loading mlp for Qwen Block %d...\n", layer_idx););
    this->gate_proj = Qwen_Linear_with_bias_Int4(
        (param_path + "/mlp/gate_proj/"),
        1, mlp_proj_dim, hidden_dim
    );
    this->up_proj   = Qwen_Linear_with_bias_Int4(
        (param_path + "/mlp/up_proj/"),
        1, mlp_proj_dim, hidden_dim
    );
    this->down_proj = Qwen_Linear_with_bias_Int4(
        (param_path + "/mlp/down_proj/"),
        1, hidden_dim, mlp_proj_dim
    );
}


Qwen3DecoderLayer_Output Qwen3DecoderLayer::forward(const Qwen3DecoderLayer_Input &input)
{
    int bs = input.hidden_states_arr.m_dim_x;
    int sqlen = input.hidden_states_arr.m_dim_y;
    int embed_dim = input.hidden_states_arr.m_dim_z;
    if (sqlen > 1)
    {
        forward_profile_name = "[ P Stage ]: " + profile_name;
    }
    else {
        forward_profile_name = "[ AG Stage ]: " + profile_name;

    }
    PROFILE_START(forward_profile_name);
    PROFILE_START(forward_profile_name + "::input layernorm");
    // -----------------------------
    // 1st stage: layernorm
    // -----------------------------
    Matrix3D<float> hidden_states = input_layernorm.forward(input.hidden_states_arr);
    PROFILE_END(forward_profile_name + "::input layernorm");

    // -----------------------------
    // 2nd stage: attention + residual addition
    // -----------------------------
    PROFILE_START(forward_profile_name + "::self-attention");
    Qwen3Attention_Input attn_param(
        hidden_states,
        input.attention_mask,
        this->layer_idx,
        input.past_sqlen
    );
    Qwen3Attention_Output attn_output = this->attn.forward(attn_param);
    Matrix3D<float> residual_out =  add(input.hidden_states_arr, attn_output.attn_output);
    PROFILE_END(forward_profile_name + "::self-attention");

    // -----------------------------
    // 3rd stage: post-attention layernorm
    // -----------------------------
    PROFILE_START(forward_profile_name + "::post attention layer norm");
    Matrix3D<float> post_attn_layernorm_out = post_attention_layernorm.forward(residual_out);
    PROFILE_END(forward_profile_name + "::post attention layer norm");

    // -----------------------------
    // 4th stage: MLP stage 
    // NOTE: This implementation differs from Qwen.cpp
    // -----------------------------
    // Gate proj: embed_dim -> hidden_dim
    PROFILE_START(forward_profile_name + "::mlp");
    PROFILE_START(forward_profile_name + "::mlp :: gate_proj");
    Matrix3D<float> gate_proj_output = gate_proj.forward(post_attn_layernorm_out);
    PROFILE_END(forward_profile_name + "::mlp :: gate_proj");
    // up proj: embed_dim -> hidden_dim
    PROFILE_START(forward_profile_name + "::mlp :: up_proj");
    Matrix3D<float> up_proj_output = up_proj.forward(post_attn_layernorm_out);
    PROFILE_END(forward_profile_name + "::mlp :: up_proj");
    // silu
    PROFILE_START(forward_profile_name + "::mlp :: SiluMul");
    gate_proj_output = Qwen3SiLuMul(gate_proj_output, up_proj_output);
    PROFILE_END(forward_profile_name + "::mlp :: SiluMul");
    // down proj: hidden_dim -> embedding
    PROFILE_START(forward_profile_name + "::mlp :: down_proj");
    Matrix3D<float> down_proj_output = down_proj.forward(gate_proj_output);
    PROFILE_END(forward_profile_name + "::mlp :: down_proj");
    PROFILE_START(forward_profile_name + "::mlp :: residual");
    residual_out = add(residual_out, down_proj_output);
    PROFILE_END(forward_profile_name + "::mlp :: residual");
    PROFILE_END(forward_profile_name + "::mlp");
    IF_DEBUG_DECODER_LAYER(
        printf("\e[31m[INFO]\e[m decoder layer statistics: \n");
        post_attn_layernorm_out.statistics();
        gate_proj_output.statistics();
        up_proj_output.statistics();
        down_proj_output.statistics();
        residual_out.statistics();
    );
    struct Qwen3DecoderLayer_Output output(residual_out, attn_output.attn_probs_reshaped);
    PROFILE_END(forward_profile_name);
    return output;
}