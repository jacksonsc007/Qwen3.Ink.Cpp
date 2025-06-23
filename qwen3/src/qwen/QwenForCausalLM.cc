#include "QwenForCausalLM.h"
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "QwenOperator.h"
#include "common.h"
#include "utils.h"

struct Qwen3ForCausalLM_Output Qwen3ForCausalLM::forward(const struct Qwen3ForCausalLM_Input& input) {
    int sqlen = input.input_ids->m_dim_z;
    if (sqlen > 1)
    {
        forward_profile_name = "[ P Stage ]: " + profile_name;
    }
    else {
        forward_profile_name = "[ AG Stage ]: " + profile_name;

    }
    PROFILE_START(forward_profile_name
    );
    // -----------------------------
    // 1st stage: setting paras and buffers
    // -----------------------------

    // -----------------------------
    // 2nd stage: evaluate decoder
    // -----------------------------
    PROFILE_START(forward_profile_name + "::decoder");
    // autoregressive generation stage
    struct Qwen3Model_Input decoder_input = {input.input_ids, past_sqlen};
    struct Qwen3Model_Output decoder_output = this->model.forward(decoder_input);
    PROFILE_END(forward_profile_name + "::decoder");

    // -----------------------------
    // 3rd stage: evaluate head to get logits
    // -----------------------------
    PROFILE_START(forward_profile_name + "::lm_head");
    int bs = decoder_output.last_hidden_state.m_dim_x;
    int h_dim = decoder_output.last_hidden_state.m_dim_z;
    ASSERT(bs == 1);
    float* last_token_last_h_ptr = &decoder_output.last_hidden_state(0, sqlen - 1, 0);
    // we only care last token output
    Matrix3D<float> last_token_last_h(last_token_last_h_ptr, 1, 1, h_dim);
    Matrix3D<float> logits = this->lm_head.forward(last_token_last_h);
    PROFILE_END(forward_profile_name + "::lm_head");
#ifdef debug_io
    // -----------------------------
    // Debug
    // -----------------------------
    std::string save_path = "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/activation/input_lm_head.bin";
    write_array_to_file(save_path.c_str(), decoder_output.last_hidden_state.m_data,
                        decoder_output.last_hidden_state.length());
    save_path = "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/activation/output_lm_head.bin";
    write_array_to_file(save_path.c_str(), logits.m_data, logits.length());
// exit(1);
#endif
    PROFILE_END(forward_profile_name);
    
    // -----------------------------
    // 4th stage: Record processed sqlen
    // -----------------------------
    int input_sqlen = input.input_ids->size();
    past_sqlen += input_sqlen;

    Qwen3ForCausalLM_Output output(std::move(logits));
    return output;
}

Qwen3ForCausalLM::Qwen3ForCausalLM(std::string param_path, const struct qwen3_config config) {
    past_sqlen = 0;
    int bs = config.batchsize;
    int h_dim = config.hidden_dim;
    int max_sqlen = config.max_sqlen;
    int vocab_size = config.vocsize;
    int num_decoder_layer = config.num_layers;
    int attn_head_dim = config.head_dim;
    int num_kv_head = config.num_kv_head;
    
    // allocate KV cahe considering maximum sequence length
    size_t cache_size = (
        num_decoder_layer * max_sqlen * attn_head_dim * num_kv_head
    );
    context_.k_cache = std::make_unique<float []>(cache_size);
    context_.v_cache = std::make_unique<float []>(cache_size);
    size_t max_weight_rows = 4096;
    size_t max_weight_cols = 151936;
    size_t num_q_blocks = max_weight_rows * max_weight_cols / QK;
    size_t size_per_q_block = (
        sizeof(uint16_t) * 4 + // scaling factors, min, zero, scaled sum
        sizeof(float) * 4 + // scaling factors, min, zero, scaled sum
        (sizeof(uint8_t) / 2) * QK + // quantized weights
        (sizeof(uint8_t) / 2) * QK // prepare for mixture of A80W40 and A81W41
    );
    size_t weight_buffer_size = (
        num_q_blocks * size_per_q_block
    );
    context_.repack_buffer = make_aligned_x86<int8_t>(64, weight_buffer_size);

    size_t max_activation_cols = 12288; // 12288 is the maximum hidden dimension
    size_t max_activation_rows = max_sqlen;
    num_q_blocks = max_activation_rows * max_activation_cols / QK;
    size_per_q_block = (
        sizeof(float) * 4 + // scaling factors, min, zero, scaled sum
        sizeof(uint16_t) * 4 + // scaling factors, min, zero, scaled sum
        (sizeof(uint8_t)) * QK 
    );

    size_t activation_buffer_size = (
        num_q_blocks * size_per_q_block
    );
    context_.activation_buffer = make_aligned_x86<int8_t>(64, activation_buffer_size);

    this->model = Qwen3Model(&context_, param_path + "/model", config);

    /*
    The linear weights are serialized from PyTorch, which has shape (out_dim, in_dim)
    */
    IF_DEBUG(printf("\e[31m[INFO]\e[m Initialize lm_head\n");)
    this->lm_head =  Qwen_Linear_with_bias_Int4(&context_, param_path + "/lm_head/", 1, vocab_size, h_dim);;
}
