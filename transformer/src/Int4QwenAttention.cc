#ifndef _Int4QwenAttention_h
#define _Int4QwenAttention_h


#include <string>
#include <sstream>
#include "Int4QwenAttention.h"
#include "operators.h"
#include "utils.h"

// TODO: 1. Why use static? 2. what if we put the static in the header?
// buffer to save QKV for current input
static float *query_states_unshape_arr; // (1 * sqlen * embed_dim)
static float *query_states_arr; // (num_head * sqlen * head_dim)
static float *key_states_unshape_arr;
static float *key_states_arr;
static float *value_states_unshape_arr;
static float *value_states_arr;
// buffer to save the KV cache
// (num_layers, 2, num_head * sqlen * head_dim)
// NOTE: The 2, in the second dim, denotes two spaces, where one for storing the current KV-cache, and the other for storing the updated expanded KV cache. 
// NOTE: 1. Lots of memory waste, cf. paged-attention of vLLM.
// TODO: try to only maintain 1 space
static float *** key_states_arr_cache; 
static float *** value_states_arr_cache;
// (num_layers, 1), indicates which space to store the updated KV cache. 
// NOTE that after we store the new KV cache, the KV cache located in the other space expires 
// and then serves as the space to store the next updated cache.
// TODO: try enum try
static int * free_space_indicator; 

static float *attn_weights_arr;
static float *attn_output_arr;
static float *attn_output_arr_reshape;
static float *attn_output_fp_arr;


//TODO: try inline
// @abstract: reshape a matrix of shape (1, sqlen, embed_dim) to shape (num_head, sqlen, head_dim)
void Int4QwenAttention::reshape_headfirst(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen)
{
    PROFILE_START("Int4QwenAttention::reshape_headfirst");
    assert(unshape.m_dim_x == 1);  // bsz == 1
    assert(unshape.m_dim_y == sqlen);
    assert(unshape.m_dim_z == this->num_head * this->head_dim);
    assert(shaped.m_dim_x == this->num_head);
    assert(shaped.m_dim_y == sqlen);
    assert(shaped.m_dim_z == this->head_dim);
    
    for (int i = 0; i < this->num_head; i++)
        for (int j = 0; j < sqlen; j++)
            for (int k = 0; k < this->head_dim; k++)
            {
                // shaped[i, j, k] = unshape[0, j, i * head_dim + k]
                shaped(i, j, k) = unshape(0, j, i * this->head_dim + k);
            }
    PROFILE_END("Int4QwenAttention::reshape_headfirst");
}

// @abstract: reshape a matrix of shape (num_head, sqlen, head_dim) to (1, sqlen, embed_dim)
void Int4QwenAttention::reshape_seqfirst(Matrix3D<float> shape1, Matrix3D<float> shape2, int sqlen)
{
    PROFILE_START("Int4QwenAttention::reshape_seqfirst");
    assert(shape2.m_dim_x == 1);  // bsz == 1
    assert(shape2.m_dim_y == sqlen);
    assert(shape2.m_dim_z == this->num_head * this->head_dim);
    assert(shape1.m_dim_x == this->num_head);
    assert(shape1.m_dim_y == sqlen);
    assert(shape1.m_dim_z == this->head_dim);
    
    for (int i = 0; i < this->num_head; i++)
        for (int j = 0; j < sqlen; j++)
            for (int k = 0; k < this->head_dim; k++)
            {
               // shape2[0, j, i * head_dim + k] = shape1[i, j, k]
               shape2(0, j, i * this->head_dim + k) = shape1(i, j, k);

            }
    PROFILE_END("Int4QwenAttention::reshape_seqfirst");
}

// TODO: memory waste check. Add static before void is wrong?
void Int4QwenAttention::initialize_memory(const struct qwen_config config) {
    allocate_aligned_memory(attn_weights_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(attn_output_fp_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(attn_output_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(attn_output_arr_reshape, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(key_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(key_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(value_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(value_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(query_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(query_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    // allocate_aligned_memory(value_states_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    free_space_indicator = new int[config.num_layers];
    for (int i = 0; i < config.num_layers; i++) free_space_indicator[i] = 0;
    // NOTE: how to allocate memory for float***
    key_states_arr_cache = new float **[config.num_layers];
    for (int i = 0; i < config.num_layers; ++i) {
        key_states_arr_cache[i] = new float *[2];
        for (int j = 0; j < 2; ++j) {
            allocate_aligned_memory(key_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(float));
        }
    }
    value_states_arr_cache = new float **[config.num_layers];
    for (int i = 0; i < config.num_layers; ++i) {
        value_states_arr_cache[i] = new float *[2];
        for (int j = 0; j < 2; ++j) {
            allocate_aligned_memory(value_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(float));
        }
    }
}

Int4QwenAttention::Int4QwenAttention(std::string param_path, const struct qwen_config config)
{
    this->q_proj =
        Qwen_Linear_with_bias_Int4( param_path + "/q_proj/", 1, config.embed_dim, config.embed_dim, 1, config.embed_dim, 1);

    this->k_proj =
        Qwen_Linear_with_bias_Int4( param_path + "/k_proj/", 1, config.embed_dim, config.embed_dim, 1, config.embed_dim, 1);

    this->v_proj =
        Qwen_Linear_with_bias_Int4( param_path + "/v_proj/", 1, config.embed_dim, config.embed_dim, 1, config.embed_dim, 1);

    this->out_proj =
        Qwen_Linear_with_bias_Int4( param_path + "/o_proj/", 1, config.embed_dim, config.embed_dim);

    // TODO: check buffer for ROPE
    float *cos_buf, *sin_buf;
    allocate_aligned_memory(cos_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    allocate_aligned_memory(sin_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    Matrix3D<float> cos(cos_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));
    Matrix3D<float> sin(sin_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));
    // this->rope_embed = RotaryPosEmb(cos, sin, param_path + "/rotary_emb");
    this->rope_embed = RotaryPosEmb(cos, sin, "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/rotary_emb");

    // NOTE: This corresponds to `sqrt(head_dim)`. Can't see the necessity of this.
    float qk_bmm_alpha;
    read_to_array((param_path + "/qk_bmm/alpha.bin").c_str(), &qk_bmm_alpha, 1);
    this->qk_bmm = BMM_F32T(qk_bmm_alpha);
    this->pv_bmm = BMM_F32T(1.0f);

    this->embed_dim = config.embed_dim;
    this->num_head = config.num_heads;
    assert(config.embed_dim % config.num_heads == 0);
    this->head_dim = config.embed_dim / config.num_heads;

}

struct Int4QwenAttention_output Int4QwenAttention::forward(const struct Int4QwenAttention_input &input)
{
    PROFILE_START(profile_name);
    struct Int4QwenAttention_output output;
    const int sqlen = input.hidden_state.m_dim_y; // the sequence length of tokens generated by last pass or the prefill tokens
    const int bs = input.hidden_state.m_dim_x;
    assert (bs == 1);
    const int layer_idx = input.layer_idx;
    
    // ---------------------------------------------------------------
    // Stage 1: Preparation of Q,K,V
    // ---------------------------------------------------------------
    // Q for current input
    Matrix3D<float> query_states_unshape(query_states_unshape_arr, bs, sqlen, embed_dim);
    this->q_proj.forward(input.hidden_state, query_states_unshape);
        // (1, sqlen, embed_dim) -> (num_head, sqlen, head_dim)
    Matrix3D<float> query_states(query_states_arr, this->num_head, sqlen, this->head_dim);
    this->reshape_headfirst(query_states_unshape, query_states, sqlen);
    
    // K for current input, needed to concat with KV Cache
    Matrix3D<float> key_states_unshape(key_states_unshape_arr, bs, sqlen, embed_dim);
    this->k_proj.forward(input.hidden_state, key_states_unshape);
        // (1, sqlen, embed_dim) -> (num_head, sqlen, head_dim)
    Matrix3D<float> key_states(key_states_arr, this->num_head, sqlen, this->head_dim);
    this->reshape_headfirst(key_states_unshape, key_states, sqlen);
    
    // V for current input, needed to concat with KV Cache
    Matrix3D<float> value_states_unshape(value_states_unshape_arr, bs, sqlen, embed_dim);
    this->v_proj.forward(input.hidden_state, value_states_unshape);
        // (1, sqlen, embed_dim) -> (num_head, sqlen, head_dim)
    Matrix3D<float> value_states(value_states_arr, this->num_head, sqlen, this->head_dim);
    this->reshape_headfirst(value_states_unshape, value_states, sqlen);
    
    
    // Add RoPE embedding TODO: read the code
    int start_idx = 0;
    if (input.has_past_key_value) start_idx = input.past_key.m_dim_y;
    this->rope_embed.forward(query_states, key_states, start_idx, sqlen);
    
    /*
    TODO: need improvement
    Concat with past key and value to get the whole context, if exits.
        KV Cache shape: (num_head, past_key_len, head_dim)
        current KV shape: (num_head, sql, head_dim)
        final KV shape: (num_head, past_key_len + sql, head_dim) 
    */
    PROFILE_START(profile_name + "::cat_past_keys_values");
    // Prepare the buffer for KV Cache to update
    float *new_value_arr_cache, *new_key_arr_cache;
    // TODO: try conditional move
    if (free_space_indicator[layer_idx] == 0)
    {
        new_key_arr_cache = key_states_arr_cache[layer_idx][0];
        new_value_arr_cache = value_states_arr_cache[layer_idx][0];
        // the original KV cache expires once we finished updating 
        free_space_indicator[layer_idx] = 1;
    }
    else
    {
        new_key_arr_cache = key_states_arr_cache[layer_idx][1];
        new_value_arr_cache = value_states_arr_cache[layer_idx][1];
        free_space_indicator[layer_idx] = 0;
    }
    
    int final_sqlen = sqlen;
    if (input.has_past_key_value)
    {
        // update KV cache
        assert (input.past_key.m_dim_x == this->num_head);
        int past_sqlen = input.past_key.m_dim_y;
        final_sqlen += past_sqlen;
        int stride_past = past_sqlen * this->head_dim;
        int stride_cur = sqlen * this->head_dim;
        float * traverse_ptr_k = new_key_arr_cache, * traverse_ptr_v = new_value_arr_cache; // only functions as index
        for (int head_idx = 0; head_idx < this->num_head; head_idx++)
        {
            // Along the head dim, append the current key and value to the cache
            memcpy(traverse_ptr_k, &input.past_key.m_data[head_idx * stride_past], sizeof(float) * stride_past);
            traverse_ptr_k += stride_past;
            memcpy(traverse_ptr_k, &key_states.m_data[head_idx * stride_cur], sizeof(float) * stride_cur);
            traverse_ptr_k += stride_cur;

            memcpy(traverse_ptr_v, &input.past_value.m_data[head_idx * stride_past], sizeof(float) * stride_past);
            traverse_ptr_v += stride_past;
            memcpy(traverse_ptr_v, &value_states.m_data[head_idx * stride_cur], sizeof(float) * stride_cur);
            traverse_ptr_v += stride_cur;
        }

    }
    else
    {
        // generate first KV Cache
        memcpy(new_key_arr_cache, key_states.m_data, sizeof(float) * this->num_head * sqlen * this->head_dim);
        memcpy(new_value_arr_cache, value_states.m_data, sizeof(float) * this->num_head * sqlen * this->head_dim);
        
    }
    // get the whole-context KV
    float *all_key_state_arr = new_key_arr_cache;
    float *all_value_state_arr = new_value_arr_cache; 

    Matrix3D<float> final_value_states(all_value_state_arr, this->num_head, final_sqlen, this->head_dim);
    Matrix3D<float> final_key_states(all_key_state_arr, this->num_head, final_sqlen, this->head_dim);
    PROFILE_END(profile_name + "::cat_past_keys_values");

    // ---------------------------------------------------------------
    // Stage 2: Attention Computation
    // ---------------------------------------------------------------
    // step1: get attention weight
    Matrix3D<float>  attn_weights(attn_weights_arr, this->num_head, sqlen, final_sqlen); // shape: (sqlen, final_sqlen)
    // (num_head, sqlen, head_dim) x (num_head, final_sqlen, head_dim) -> (num_head, sqlen, final_sqlen)
    this->qk_bmm.forward(query_states, final_key_states, attn_weights);
    // step2: apply causal mask
    batch_Add(attn_weights, input.attention_mask, attn_weights);
    for (int i = 0; i < attn_weights.length(); i++)
    {
        if (std::isinf(attn_weights.m_data[i]))
        {
            attn_weights.m_data[i] = std::numeric_limits<float>::lowest();
        }
    }
    // step3: apply softmax
    Matrix3D<float> attn_probs(attn_weights_arr, this->num_head, sqlen, final_sqlen);
    // TODO: check the softmax implementation. Why find the max value?
    softmax(attn_weights, attn_probs, 2);

    // step4: get output
    Matrix3D<float> attn_output(attn_output_arr, this->num_head, sqlen, this->head_dim);
    // TODO: there is a legacy implementation, need to check
    this->pv_bmm.forward_weight_untransposed(attn_probs, final_value_states, attn_output);
    // step5: reshape output: (num_head, sqlen, head_dim) -> (1, sqlen, sqlen * head_dim)
    Matrix3D<float> attn_reshape(attn_output_arr_reshape, bs, sqlen, this->num_head * this->head_dim);
    this->reshape_seqfirst(attn_output, attn_reshape, sqlen);
    // step6: output projection
    Matrix3D<float> attn_output_fp(attn_output_fp_arr, bs, sqlen, this->embed_dim);
    this->out_proj.forward(attn_reshape, attn_output_fp);

    // --------------------------------------------------------------
    // Debug
    // --------------------------------------------------------------
#if DEBUG_INT4
    query_states.statistics();
    key_states.statistics();
    value_states.statistics();
    attn_output.statistics();
    attn_output_fp.statistics();
#endif
    

#if DEBUG == 1
    
    std::ostringstream oss;
    std::string save_path;
    // input
    oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer" << input.layer_idx << 
        "/self_attn/activation/input.bin";
    save_path = oss.str();
    write_array_to_file(save_path.c_str(), input.hidden_state.m_data, input.hidden_state.length());
    // q,k,v
    oss.str("");
    oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer" << input.layer_idx << 
        "/self_attn/activation/q.bin";
    save_path = oss.str();
    write_array_to_file(save_path.c_str(), query_states_unshape.m_data, query_states_unshape.length());
    oss.str("");
    oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer" << input.layer_idx << 
        "/self_attn/activation/k.bin";
    save_path = oss.str();
    write_array_to_file(save_path.c_str(), key_states_unshape.m_data, key_states_unshape.length());
    oss.str("");
    oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer" << input.layer_idx << 
        "/self_attn/activation/v.bin";
    save_path = oss.str();
    write_array_to_file(save_path.c_str(), value_states_unshape.m_data, value_states_unshape.length());

    // q,k fed with rope
    oss.str(""); 
    oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer" << input.layer_idx << 
        "/self_attn/activation/q_with_rope.bin";
    save_path = oss.str();
    write_array_to_file(save_path.c_str(), query_states.m_data, query_states.length());
    oss.str(""); 
    oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer" << input.layer_idx << 
        "/self_attn/activation/k_with_rope.bin";
    save_path = oss.str();
    write_array_to_file(save_path.c_str(), key_states.m_data, key_states.length());
    
    // final q,k,v
    oss.str(""); 
    oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer" << input.layer_idx << 
        "/self_attn/activation/final_q.bin";
    save_path = oss.str();
    write_array_to_file(save_path.c_str(), query_states.m_data, query_states.length());
    oss.str(""); 
    oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer" << input.layer_idx << 
        "/self_attn/activation/final_k.bin";
    save_path = oss.str();
    write_array_to_file(save_path.c_str(), final_key_states.m_data, final_key_states.length());
    oss.str(""); 
    oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer" << input.layer_idx << 
        "/self_attn/activation/final_v.bin";
    save_path = oss.str();
    write_array_to_file(save_path.c_str(), final_value_states.m_data, final_value_states.length());

    // attention output before out_proj
    oss.str("");
    oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer" << input.layer_idx << 
        "/self_attn/activation/attn_before_out.bin";
    save_path = oss.str();
    write_array_to_file(save_path.c_str(), attn_reshape.m_data, attn_reshape.length());
#endif

    // ---------------------------------------------------------------
    // stage3: Output assignment and wrapping up instrumentation code
    // ---------------------------------------------------------------
    output.attn_output = attn_output_fp;
    output.past_key_value = { final_key_states, final_value_states };
    PROFILE_END(profile_name);
    return output;
}

#endif