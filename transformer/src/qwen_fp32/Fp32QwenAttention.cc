#ifndef _Fp32QwenAttention_h
#define _Fp32QwenAttention_h


#include <string>
#include <sstream>
#include "Fp32QwenAttention.h"
#include "common.h"
#include "operators.h"
#include "utils.h"

// TODO: 1. Why use static? 2. what if we put the static in the header?
// buffer to save QKV for current input
static float *query_states_unshape_arr; // (1 * sqlen * embed_dim)
static float *query_states_norm_arr; // (1 * sqlen * embed_dim)
static float *query_states_arr; // (num_head * sqlen * head_dim)
static float *key_states_unshape_arr;
static float *key_states_norm_arr;
static float *key_states_arr;
static float *value_states_unshape_arr;
static float *value_states_arr;

// GQA
static float *key_states_arr_expanded;
static float *value_states_arr_expanded;
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
static float *attn_probs_arr;
static float *attn_output_arr;
static float *attn_output_arr_reshape;
static float *attn_output_fp_arr;


// TODO: memory waste check. Add static before void is wrong?
void Fp32Qwen3Attention::initialize_memory(const struct qwen3_config config) {
    int max_sqlen = config.max_sqlen;
    int hidden_dim = config.hidden_dim;
    int num_heads = config.num_q_head;
    int num_layers = config.num_layers;
    allocate_aligned_memory(attn_weights_arr, num_heads * max_sqlen * max_sqlen * sizeof(float));
    allocate_aligned_memory(attn_probs_arr, num_heads * max_sqlen * max_sqlen * sizeof(float));
    allocate_aligned_memory(attn_output_fp_arr, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(attn_output_arr, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(attn_output_arr_reshape, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(key_states_unshape_arr, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(key_states_arr, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(value_states_unshape_arr, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(value_states_arr, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(query_states_arr, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(query_states_unshape_arr, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(query_states_norm_arr, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(key_states_norm_arr, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(key_states_arr_expanded, max_sqlen * hidden_dim * sizeof(float));
    allocate_aligned_memory(value_states_arr_expanded, max_sqlen * hidden_dim * sizeof(float));

    // allocate_aligned_memory(value_states_transpose_arr, max_sqlen * hidden_dim * sizeof(float));
    free_space_indicator = new int[num_layers];
    for (int i = 0; i < num_layers; i++) free_space_indicator[i] = 0;
    // NOTE: how to allocate memory for float***
    key_states_arr_cache = new float **[num_layers];
    for (int i = 0; i < num_layers; ++i) {
        key_states_arr_cache[i] = new float *[2];
        for (int j = 0; j < 2; ++j) {
            allocate_aligned_memory(key_states_arr_cache[i][j], max_sqlen * hidden_dim * sizeof(float));
        }
    }
    value_states_arr_cache = new float **[num_layers];
    for (int i = 0; i < num_layers; ++i) {
        value_states_arr_cache[i] = new float *[2];
        for (int j = 0; j < 2; ++j) {
            allocate_aligned_memory(value_states_arr_cache[i][j], max_sqlen * hidden_dim * sizeof(float));
        }
    }
}

Fp32Qwen3Attention::Fp32Qwen3Attention(std::string param_path, const struct qwen3_config config)
{
    max_sqlen = config.max_sqlen;
    hidden_dim = config.hidden_dim;
    q_dim = hidden_dim;
    kv_dim = hidden_dim / 4;
    num_q_head = config.num_q_head;
    num_kv_head = config.num_k_head;
    head_dim = q_dim / num_q_head;
    assert (head_dim == kv_dim / num_kv_head);

    float *q_weight, *k_weight, *v_weight, *out_weight;
    allocate_aligned_memory(q_weight, hidden_dim * hidden_dim * sizeof(float));
    allocate_aligned_memory(k_weight, hidden_dim * hidden_dim * sizeof(float));
    allocate_aligned_memory(v_weight, hidden_dim * hidden_dim * sizeof(float));
    allocate_aligned_memory(out_weight, hidden_dim * hidden_dim * sizeof(float));

    // TODO: check the correspondence btw weight name and offline weight file.
    IF_DEBUG(
        printf("\e[31m[INFO]\e[m Loading Attention Module\n");
    );
    q_proj =
        LinearFp32(
            Matrix3D<float>(q_weight, 1, q_dim, hidden_dim), param_path + "/q_proj/weight.bin"
        );
    k_proj =
        LinearFp32(
            Matrix3D<float>(k_weight, 1, kv_dim, hidden_dim), param_path + "/k_proj/weight.bin"
        );
    v_proj =
        LinearFp32(
            Matrix3D<float>(v_weight, 1, kv_dim, hidden_dim), param_path + "/v_proj/weight.bin"
        );
    o_proj =
        LinearFp32(Matrix3D<float>(out_weight, 1, hidden_dim, hidden_dim), param_path + "/o_proj/weight.bin");
    IF_DEBUG(
        printf("\e[31m[INFO]\e[m Loading Attention qk norm\n");
    );
    float *q_norm_weight;
    allocate_aligned_memory(q_norm_weight, head_dim * sizeof(float));
    Matrix3D<float> q_norm_mat(q_norm_weight, 1, 1, head_dim);
    q_norm_mat.load((param_path + "/q_norm/weight.bin").c_str());
    q_norm = Qwen3RMSNorm(q_norm_mat);

    float *k_norm_weight;
    allocate_aligned_memory(k_norm_weight, head_dim * sizeof(float));
    Matrix3D<float> k_norm_mat(k_norm_weight, 1, 1, head_dim);
    k_norm_mat.load((param_path + "/k_norm/weight.bin").c_str());
    k_norm = Qwen3RMSNorm(k_norm_mat);

    // ROPE
    float *cos_buf, *sin_buf;
    allocate_aligned_memory(cos_buf, config.max_sqlen * head_dim * sizeof(float));
    allocate_aligned_memory(sin_buf, config.max_sqlen * head_dim * sizeof(float));
    Matrix3D<float> cos(cos_buf, 1, config.max_sqlen, head_dim);
    Matrix3D<float> sin(sin_buf, 1, config.max_sqlen, head_dim);
    this->rope_embed = RotaryPosEmb(cos, sin, param_path + "/../../../rotary_emb");

    // scaling factor
    float qk_bmm_alpha;
    read_to_array((param_path + "/scaling.bin").c_str(), &qk_bmm_alpha, 1);
    this->qk_bmm = BMM_F32T(qk_bmm_alpha);
    this->pv_bmm = BMM_F32T(1.0f);


}

struct Fp32Qwen3Attention_Output Fp32Qwen3Attention::forward(const struct Fp32Qwen3Attention_Input &input)
{
    PROFILE_START(profile_name);
    struct Fp32Qwen3Attention_Output output;
    const int sqlen = input.hidden_state.m_dim_y; // the sequence length of tokens generated by last pass or the prefill tokens
    const int bs = input.hidden_state.m_dim_x;
    const int layer_idx = input.layer_idx;
    
    // ---------------------------------------------------------------
    // Stage 1: Generation of Q,K,V
    // ---------------------------------------------------------------
    // TODO: Fused Generation of QKV
    assert (bs == 1);
    Matrix3D<float> query_states_unshape(query_states_unshape_arr, bs, sqlen, q_dim);
    Matrix3D<float> query_states_norm(query_states_norm_arr, bs * sqlen, num_q_head, head_dim);
    Matrix3D<float> query_states(query_states_arr, num_q_head, bs * sqlen, head_dim);
    this->q_proj.forward(input.hidden_state, query_states_unshape);
    query_states_unshape.view(bs * sqlen, num_q_head, head_dim);
    this->q_norm.forward(query_states_unshape, query_states_norm);
    // (bs * sqlen, n_head, head_dim) -> (num_head, bs*sqlen, head_dim)
    permute01(query_states_norm, query_states);
    
    Matrix3D<float> key_states_unshape(key_states_unshape_arr, bs, sqlen, kv_dim);
    Matrix3D<float> key_states_norm(key_states_norm_arr, bs * sqlen, num_kv_head, head_dim);
    Matrix3D<float> key_states(key_states_arr, num_kv_head, bs * sqlen, head_dim);
    this->k_proj.forward(input.hidden_state, key_states_unshape);
    key_states_unshape.view(bs * sqlen, num_kv_head, head_dim);
    this->k_norm.forward(key_states_unshape, key_states_norm);
    permute01(key_states_norm, key_states);
    
    Matrix3D<float> value_states_unshape(value_states_unshape_arr, bs, sqlen, kv_dim);
    Matrix3D<float> value_states(value_states_arr, num_kv_head, sqlen, this->head_dim);
    this->v_proj.forward(input.hidden_state, value_states_unshape);
    value_states_unshape.view(bs * sqlen, num_kv_head, head_dim);
    permute01(value_states_unshape, value_states);
    
    int start_idx = 0;
    if (input.has_past_key_value) start_idx = input.past_key.m_dim_y;
    this->rope_embed.forward(query_states, key_states, start_idx, sqlen);
    
    /*
    TODO: need improvement
    Refresh KV Cache
        KV Cache shape  : ( num_head, past_key_len          , head_dim ) 
        current KV shape: ( num_head, cur_sql               , head_dim ) 
        final KV shape  : ( num_head, past_key_len + cur_sql, head_dim) 
    In this poor implementation, new KV cache space is managed by copying original KV cache and current KV value.
    Why do we do this? The reason is that the old KV cache takes up a chunk of contiguous space, we need a new space
    when append new elements along the second dimension.
    free_space_indicator identifies new and old KV cache space.
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
    
    int total_context_sqlen = sqlen;
    if (input.has_past_key_value)
    {
        assert (sqlen == 1);
        // update KV cache
        assert (input.past_key.m_dim_x == num_kv_head);
        int past_sqlen = input.past_key.m_dim_y;
        total_context_sqlen += past_sqlen;

        // (num_heads, past_len, head_dim) -> (num_heads, past_len + cur_len, head_dim); Row-major
        int num_past_element = past_sqlen * head_dim;
        int num_cur_element = sqlen * head_dim;
        float * traverse_ptr_k = new_key_arr_cache, * traverse_ptr_v = new_value_arr_cache; // only functions as index
        for (int head_idx = 0; head_idx < num_kv_head; head_idx++)
        {
            // Along the head dim, append the current key and value to the cache
            memcpy(traverse_ptr_k, &input.past_key.m_data[head_idx * num_past_element], sizeof(float) * num_past_element);
            traverse_ptr_k += num_past_element;
            memcpy(traverse_ptr_k, &key_states.m_data[head_idx * num_cur_element], sizeof(float) * num_cur_element);
            traverse_ptr_k += num_cur_element;

            memcpy(traverse_ptr_v, &input.past_value.m_data[head_idx * num_past_element], sizeof(float) * num_past_element);
            traverse_ptr_v += num_past_element;
            memcpy(traverse_ptr_v, &value_states.m_data[head_idx * num_cur_element], sizeof(float) * num_cur_element);
            traverse_ptr_v += num_cur_element;
        }

    }
    else
    {
        // generate first KV Cache
        memcpy(new_key_arr_cache, key_states.m_data, sizeof(float) * num_kv_head * sqlen * head_dim);
        memcpy(new_value_arr_cache, value_states.m_data, sizeof(float) * num_kv_head * sqlen * head_dim);
        
    }
    // get the whole-context KV
    float *all_key_state_arr = new_key_arr_cache;
    float *all_value_state_arr = new_value_arr_cache; 

    int group_size = num_q_head / num_kv_head;
    Matrix3D<float> final_value_states(all_value_state_arr, num_kv_head, total_context_sqlen, head_dim);
    Matrix3D<float> final_value_states_expanded = final_value_states.repeat(0, group_size, value_states_arr_expanded);
    Matrix3D<float> final_key_states(all_key_state_arr, num_kv_head, total_context_sqlen, head_dim);
    Matrix3D<float> final_key_states_expanded = final_key_states.repeat(0, group_size, key_states_arr_expanded);

    PROFILE_END(profile_name + "::cat_past_keys_values");

    // ---------------------------------------------------------------
    // Stage 2: Attention Computation
    // ---------------------------------------------------------------
    // step1: get attention weight
    Matrix3D<float>  attn_weights(attn_weights_arr, num_q_head, sqlen, total_context_sqlen); // shape: (sqlen, final_sqlen)
    // (num_head, sqlen, head_dim) x (num_head, final_sqlen, head_dim) -> (num_head, sqlen, final_sqlen)
    this->qk_bmm.forward(query_states, final_key_states_expanded, attn_weights);
    assert(not has_nan(attn_weights));
    IF_DEBUG_ATTENTION(        
        query_states.statistics();
        final_key_states_expanded.statistics();
        attn_weights.statistics();
    );
    // step2: apply causal mask
    batch_Add(attn_weights, input.attention_mask, attn_weights);
    assert(not has_nan(attn_weights));
    for (int i = 0; i < attn_weights.length(); i++)
    {
        if (std::isinf(attn_weights.m_data[i]))
        {
            attn_weights.m_data[i] = std::numeric_limits<float>::lowest();
        }
    }
    assert(not has_nan(attn_weights));
    // step3: apply softmax
    Matrix3D<float> attn_probs(attn_probs_arr, num_q_head, sqlen, total_context_sqlen);
    // TODO: check the softmax implementation. Why find the max value?
    softmax(attn_weights, attn_probs, 2);
    
    assert(not has_nan(attn_probs));

    // step4: get output
    Matrix3D<float> attn_output(attn_output_arr, num_q_head, bs * sqlen, head_dim);
    // TODO: there is a legacy implementation, need to check
    this->pv_bmm.forward_weight_untransposed(attn_probs, final_value_states_expanded, attn_output);
    // step5: reshape output: (num_head, sqlen, head_dim) -> (1, sqlen, sqlen * head_dim)
    Matrix3D<float> attn_reshape(attn_output_arr_reshape, bs * sqlen, num_q_head, head_dim);
    permute01(attn_output, attn_reshape);
    attn_reshape.view(bs, sqlen, hidden_dim);
    // step6: output projection
    Matrix3D<float> attn_output_fp(attn_output_fp_arr, bs, sqlen, hidden_dim);
    this->o_proj.forward(attn_reshape, attn_output_fp);

    // --------------------------------------------------------------
    // Debug
    // --------------------------------------------------------------
    IF_DEBUG_ATTENTION(        
        query_states.statistics();
        key_states.statistics();
        value_states.statistics();
        attn_output.statistics();
        attn_output_fp.statistics();
    );
    IF_DEBUG_IO(

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
    );
    

    // ---------------------------------------------------------------
    // stage3: Output assignment and wrapping up instrumentation code
    // ---------------------------------------------------------------
    output.attn_output = attn_output_fp;
    output.past_key_value = { final_key_states, final_value_states };
    PROFILE_END(profile_name);
    return output;
}

#endif