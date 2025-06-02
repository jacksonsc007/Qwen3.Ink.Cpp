#include "Int4QwenDecoderLayer.h"

// TODO: why static? Could we reuse the same space to hold the following variables?
static float* hidden_states_arr;
static float* hidden_states_res_arr;
static float* post_attn_layernorm_out_arr;
static float *gate_proj_arr;
static float *up_proj_arr;
static float *down_proj_arr;

// TODO: check this silu
void QwenInt4SiLuMul(Matrix3D<float> a, Matrix3D<float> b) {
    PROFILE_START("MulSiLu");
    for (int i = 0; i < a.length(); i++) {
        float v = a.m_data[i];
        float silu_v = v * (1.0 / (1.0 + exp(-1 * v)));
        a.m_data[i] = silu_v * b.m_data[i];
    }
    PROFILE_END("MulSiLu");
}

template <typename T>
void add(Matrix3D<T> a, Matrix3D<T> b, Matrix3D<T> c) {
    PROFILE_START("Int4QwenDecoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("Int4QwenDecoderLayer::add");
}

Int4QwenBlock::Int4QwenBlock(std::string param_path, const struct qwen_config config, int layer_idx)
{
    int max_sqlen_ = config.max_sqlen;
    int embed_dim_ = config.embed_dim;
    int hidden_dim_ = config.hidden_dim;
    int num_heads_ = config.num_heads;
    if (layer_idx == 0)
    {
        // TODO: Shoud me free there memory maunally?
        allocate_aligned_memory(hidden_states_arr, max_sqlen_ * embed_dim_ * sizeof(float));
        allocate_aligned_memory(hidden_states_res_arr, max_sqlen_ * embed_dim_ * sizeof(float));
        allocate_aligned_memory(post_attn_layernorm_out_arr, max_sqlen_ * embed_dim_ * sizeof(float));
        allocate_aligned_memory(gate_proj_arr, max_sqlen_ * embed_dim_ * sizeof(float));
        allocate_aligned_memory(down_proj_arr, max_sqlen_ * embed_dim_ * sizeof(float));
        allocate_aligned_memory(up_proj_arr, max_sqlen_ * embed_dim_ * sizeof(float));
        // intialize attention layer
        Int4QwenAttention::initialize_memory(config);
    }
    
    float *ln_1_weight;
    allocate_aligned_memory(ln_1_weight, embed_dim_ * sizeof(float));
    Matrix3D<float> ln_1_weight_mat(ln_1_weight, 1, 1, embed_dim_);
    ln_1_weight_mat.load((param_path + "/ln_1/weight.bin").c_str());
    this->ln_1 = QwenInt4RMSNorm(ln_1_weight_mat);
    float *ln_2_weight;
    allocate_aligned_memory(ln_2_weight, embed_dim_ * sizeof(float));
    Matrix3D<float> ln_2_weight_mat(ln_2_weight, 1, 1, embed_dim_);
    ln_2_weight_mat.load((param_path + "/ln_2/weight.bin").c_str());
    this->ln_2 = QwenInt4RMSNorm(ln_2_weight_mat);
    this->embed_dim = embed_dim_;
    this->hidden_dim = hidden_dim_;
    this->num_attention_heads = num_heads_;
    this->layer_idx = layer_idx;
    this->attn = Int4QwenAttention(param_path + "/self_attn", config);

    this->gate_proj = Qwen_Linear_with_bias_Int4(param_path + "/w2/", 1, hidden_dim, embed_dim_);
    this->up_proj = Qwen_Linear_with_bias_Int4(param_path + "/w1/", 1, hidden_dim_, embed_dim_);
    this->down_proj = Qwen_Linear_with_bias_Int4(param_path + "/c_proj/", 1, embed_dim_, hidden_dim_);
}


Int4QwenBlock_output Int4QwenBlock::forward(const Int4QwenBlock_input &input)
{
    PROFILE_START(profile_name);
    // -----------------------------
    // 1st stage: layernorm
    // -----------------------------
    PROFILE_START(profile_name + ":: ln_1");
    int bs = input.hidden_states_arr.m_dim_x;
    int sq_len = input.hidden_states_arr.m_dim_y;
    int embed_dim = input.hidden_states_arr.m_dim_z;
    Matrix3D<float> hidden_states(hidden_states_arr, bs, sq_len, embed_dim);
    this->ln_1.forward(input.hidden_states_arr, hidden_states);
    PROFILE_END(profile_name + ":: ln_1");
    // -----------------------------
    // 2nd stage: attention + residual addition
    // -----------------------------
    PROFILE_START(profile_name + ":: attention");
    Int4QwenAttention_input attn_param(
        hidden_states, input.attention_mask, input.past_key,
        input.past_value, input.has_past_key_value, this->layer_idx
    );
    Int4QwenAttention_output attn_output = this->attn.forward(attn_param);
    Matrix3D<float> residual_out(hidden_states_res_arr, bs, sq_len, embed_dim);
    add(input.hidden_states_arr, attn_output.attn_output, residual_out);
    PROFILE_END(profile_name + ":: attention");
    // -----------------------------
    // 3rd stage: post-attention layernorm
    // -----------------------------
    PROFILE_START(profile_name + ":: ln_2");
    Matrix3D<float> post_attn_layernorm_out(post_attn_layernorm_out_arr, bs, sq_len, embed_dim);
    this->ln_2.forward(residual_out, post_attn_layernorm_out);
    PROFILE_END(profile_name + ":: ln_2");
    // -----------------------------
    // 4th stage: MLP stage 
    // NOTE: This implementation differs from Qwen.cpp
    // -----------------------------
    // Gate proj: embed_dim -> hidden_dim
    PROFILE_START(profile_name + ":: mlp");
    Matrix3D<float> gate_proj(gate_proj_arr, bs, sq_len, this->hidden_dim);
    this->gate_proj.forward(post_attn_layernorm_out, gate_proj);
    // up proj: embed_dim -> hidden_dim
    Matrix3D<float> up_proj(up_proj_arr, bs, sq_len, this->hidden_dim);
    this->up_proj.forward(post_attn_layernorm_out, up_proj);
    // silu
    QwenInt4SiLuMul(gate_proj, up_proj);
    // down proj: hidden_dim -> embedding
    Matrix3D<float> down_proj(down_proj_arr, bs, sq_len, this->embed_dim);
    this->down_proj.forward(gate_proj, down_proj);
    // TODO: performance change if we allocate another space for the second residual output?
    add(residual_out, down_proj, residual_out);
    PROFILE_END(profile_name + ":: mlp");

    struct Int4QwenBlock_output output(residual_out, attn_output.attn_probs_reshaped,
                                               attn_output.past_key_value);
    // //////////////////
    // Debug
#if DEBUG_INT4
    printf("\e[31m[INFO]\e[m decoder layer statistics: \n");
    residual_out.statistics();
    post_attn_layernorm_out.statistics();
    gate_proj.statistics();
    up_proj.statistics();
    down_proj.statistics();
    residual_out.statistics();
#endif
    // //////////////////
    PROFILE_END(profile_name);
    return output;
}