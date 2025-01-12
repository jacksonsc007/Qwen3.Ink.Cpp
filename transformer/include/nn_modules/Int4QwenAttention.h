#include "common.h"
#include "operators.h"
#include "Int4QwenOperator.h"


struct Int4QwenAttention_output {
    Matrix3D<float> attn_output;
    Matrix3D<float> attn_probs_reshaped;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;
};

struct Int4QwenAttention_input
{
    Matrix3D<float> hidden_state; //(bs, sqlen, embed_dim)
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key;
    Matrix3D<float> past_value;
    bool has_past_key_value = false;
    int layer_idx;
    
    Int4QwenAttention_input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, int layer_idx_)
        : hidden_state(hidden_states_), attention_mask(attention_mask_), layer_idx(layer_idx_) {}

    Int4QwenAttention_input(
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


class Int4QwenAttention {
public:
    // construct
    Int4QwenAttention(){}
    Int4QwenAttention(std::string param_path, const struct qwen_config config);
    // member
    int num_head, embed_dim, head_dim;
    Qwen_Linear_with_bias_Int4 k_proj, v_proj, q_proj;
    Qwen_Linear_with_bias_Int4 out_proj;
    RotaryPosEmb rope_embed;
    BMM_F32T qk_bmm, pv_bmm;
    std::string profile_name = "Int4QwenAttention";
    // method
    static void initialize_memory(const struct qwen_config config);
    struct Int4QwenAttention_output forward(const struct Int4QwenAttention_input &input);
    void reshape_headfirst(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen);
    void reshape_seqfirst(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen);
};