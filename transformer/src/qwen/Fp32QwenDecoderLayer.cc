#include "common.h"
#include "Fp32QwenDecoderLayer.h"

// TODO: why static? Could we reuse the same space to hold the following variables?
static float* hidden_states_arr;
static float* hidden_states_res_arr;
static float* post_attn_layernorm_out_arr;
static float *gate_proj_arr;
static float *up_proj_arr;
static float *down_proj_arr;

// TODO: check this silu
void Qwen3SiLuMul(Matrix3D<float> a, Matrix3D<float> b) {
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
    PROFILE_START("Fp32QwenDecoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("Fp32QwenDecoderLayer::add");
}

Fp32Qwen3DecoderLayer::Fp32Qwen3DecoderLayer(std::string param_path, const struct qwen3_config config, int layer_idx)
{
    this->layer_idx = layer_idx;
    max_sqlen = config.max_sqlen;
    hidden_dim = config.hidden_dim;
    if (layer_idx == 0)
    {
        // TODO: Shoud me free there memory maunally?
        allocate_aligned_memory(hidden_states_arr, max_sqlen * hidden_dim * sizeof(float));
        allocate_aligned_memory(hidden_states_res_arr, max_sqlen * hidden_dim * sizeof(float));
        allocate_aligned_memory(post_attn_layernorm_out_arr, max_sqlen * hidden_dim * sizeof(float));
        allocate_aligned_memory(gate_proj_arr, max_sqlen * hidden_dim * sizeof(float));
        allocate_aligned_memory(down_proj_arr, max_sqlen * hidden_dim * sizeof(float));
        allocate_aligned_memory(up_proj_arr, max_sqlen * hidden_dim * sizeof(float));
        // intialize attention layer
        Fp32Qwen3Attention::initialize_memory(config);
    }
    
    // input layernorm
    float *ln_1_weight;
    allocate_aligned_memory(ln_1_weight, hidden_dim * sizeof(float));
    Matrix3D<float> ln_1_weight_mat(ln_1_weight, 1, 1, hidden_dim);
    IF_DEBUG(
        printf("\e[31m[INFO]\e[m Loading input layernorm for Qwen Block %d...\n", layer_idx);
    );
    ln_1_weight_mat.load((param_path + "/input_layernorm/weight.bin").c_str());
    this->input_layernorm = Qwen3RMSNorm(ln_1_weight_mat);

    // post attention layernorm
    float *ln_2_weight;
    allocate_aligned_memory(ln_2_weight, hidden_dim * sizeof(float));
    Matrix3D<float> ln_2_weight_mat(ln_2_weight, 1, 1, hidden_dim);
    IF_DEBUG(
        printf("\e[31m[INFO]\e[m Loading post-attention layernorm for Qwen Block %d...\n", layer_idx);
    );
    ln_2_weight_mat.load((param_path + "/post_attention_layernorm/weight.bin").c_str());
    this->post_attention_layernorm = Qwen3RMSNorm(ln_2_weight_mat);

    // attention module
    this->attn = Fp32Qwen3Attention(param_path + "/self_attn", config);

    // mlp module
    float *gate_proj_weight, *down_proj_weight, *up_proj_weight;
    allocate_aligned_memory(gate_proj_weight, hidden_dim * mlp_proj_dim * sizeof(float));
    allocate_aligned_memory(up_proj_weight, hidden_dim * mlp_proj_dim * sizeof(float));
    allocate_aligned_memory(down_proj_weight, mlp_proj_dim * hidden_dim * sizeof(float));
    IF_DEBUG(
        printf("\e[31m[INFO]\e[m Loading mlp for Qwen Block %d...\n", layer_idx);
    );
    this->gate_proj = LinearFp32(
        Matrix3D<float>(gate_proj_weight, 1, mlp_proj_dim, hidden_dim),
        (param_path + "/mlp/gate_proj/weight.bin")
    );
    this->up_proj   = LinearFp32(
        Matrix3D<float>(up_proj_weight, 1, mlp_proj_dim, hidden_dim),
        (param_path + "/mlp/up_proj/weight.bin")
    );
    this->down_proj = LinearFp32(
        Matrix3D<float>(down_proj_weight, 1, hidden_dim, mlp_proj_dim),
        (param_path + "/mlp/down_proj/weight.bin")
    );
}


Fp32Qwen3DecoderLayer_Output Fp32Qwen3DecoderLayer::forward(const Fp32Qwen3DecoderLayer_Input &input)
{
    PROFILE_START(profile_name);
    // -----------------------------
    // 1st stage: layernorm
    // -----------------------------
    int bs = input.hidden_states_arr.m_dim_x;
    int sq_len = input.hidden_states_arr.m_dim_y;
    int embed_dim = input.hidden_states_arr.m_dim_z;
    Matrix3D<float> hidden_states(hidden_states_arr, bs, sq_len, embed_dim);
    this->input_layernorm.forward(input.hidden_states_arr, hidden_states);

    // -----------------------------
    // 2nd stage: attention + residual addition
    // -----------------------------
    Fp32Qwen3Attention_Input attn_param(
        hidden_states, input.attention_mask, input.past_key,
        input.past_value, input.has_past_key_value, this->layer_idx
    );
    Fp32Qwen3Attention_Output attn_output = this->attn.forward(attn_param);
    Matrix3D<float> residual_out(hidden_states_res_arr, bs, sq_len, embed_dim);
    add(input.hidden_states_arr, attn_output.attn_output, residual_out);

    // -----------------------------
    // 3rd stage: post-attention layernorm
    // -----------------------------
    Matrix3D<float> post_attn_layernorm_out(post_attn_layernorm_out_arr, bs, sq_len, embed_dim);
    this->post_attention_layernorm.forward(residual_out, post_attn_layernorm_out);

    // -----------------------------
    // 4th stage: MLP stage 
    // NOTE: This implementation differs from Qwen.cpp
    // -----------------------------
    // Gate proj: embed_dim -> hidden_dim
    Matrix3D<float> gate_proj_output(gate_proj_arr, bs, sq_len, mlp_proj_dim);
    this->gate_proj.forward(post_attn_layernorm_out, gate_proj_output);
    // up proj: embed_dim -> hidden_dim
    Matrix3D<float> up_proj_output(up_proj_arr, bs, sq_len, mlp_proj_dim);
    this->up_proj.forward(post_attn_layernorm_out, up_proj_output);
    // silu
    Qwen3SiLuMul(gate_proj_output, up_proj_output);
    // down proj: hidden_dim -> embedding
    Matrix3D<float> down_proj_output(down_proj_arr, bs, sq_len, hidden_dim);
    this->down_proj.forward(gate_proj_output, down_proj_output);
    add(residual_out, down_proj_output, residual_out);
    
    IF_DEBUG_DECODER_LAYER(
        printf("\e[31m[INFO]\e[m decoder layer statistics: \n");
        post_attn_layernorm_out.statistics();
        gate_proj_output.statistics();
        up_proj_output.statistics();
        down_proj_output.statistics();
        residual_out.statistics();
    );
    struct Fp32Qwen3DecoderLayer_Output output(residual_out, attn_output.attn_probs_reshaped,
                                               attn_output.past_key_value);
    PROFILE_END(profile_name);
    return output;
}