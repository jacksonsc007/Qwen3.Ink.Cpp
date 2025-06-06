#include "QwenForCausalLM.h"

#include "QwenOperator.h"
#include "common.h"
#include "utils.h"

struct Qwen3ForCausalLM_Output Qwen3ForCausalLM::forward(const struct Qwen3ForCausalLM_Input& input) {
    PROFILE_START(profile_name);
    // -----------------------------
    // 1st stage: setting paras and buffers
    // -----------------------------
    struct Qwen3Model_Output decoder_output;

    // -----------------------------
    // 2nd stage: evaluate decoder
    // -----------------------------
    PROFILE_START(profile_name + "::decoder");
    if (input.has_past_keys_values) {
        // autoregressive generation stage
        struct Qwen3Model_Input decoder_input = {input.input_ids, input.past_keys, input.past_values};
        decoder_output = this->model.forward(decoder_input);
    } else {
        // prompt stage
        struct Qwen3Model_Input decoder_input = {input.input_ids};
        decoder_output = this->model.forward(decoder_input);
    }
    PROFILE_END(profile_name + "::decoder");

    // -----------------------------
    // 3rd stage: evaluate head to get logits
    // -----------------------------
    PROFILE_START(profile_name + "::lm_head");
    int bs = decoder_output.last_hidden_state.m_dim_x;
    int sqlen = decoder_output.last_hidden_state.m_dim_y;
    int h_dim = decoder_output.last_hidden_state.m_dim_z;
    ASSERT(bs == 1);

    float* last_token_last_h_ptr = &decoder_output.last_hidden_state(0, sqlen - 1, 0);
    Matrix3D<float> last_token_last_h(last_token_last_h_ptr, 1, sqlen, h_dim);
    Matrix3D<float> logits = this->lm_head.forward(last_token_last_h);
    PROFILE_END(profile_name + "::lm_head");

    Qwen3ForCausalLM_Output output = {logits, decoder_output.past_keys, decoder_output.past_values};

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
    PROFILE_END(profile_name);
    return output;
}

Qwen3ForCausalLM::Qwen3ForCausalLM(std::string param_path, const struct qwen3_config config) {
    int bs = config.batchsize;
    int h_dim = config.hidden_dim;
    int max_sqlen = config.max_sqlen;
    int vocab_size = config.vocsize;
    lm_head_weight = Matrix3D<float>(1, h_dim, vocab_size);

    this->model = Qwen3Model(param_path + "/model", config);

    /*
    The linear weights are serialized from PyTorch, which has shape (out_dim, in_dim)
    */
    this->lm_head = Qwen_Linear_with_bias_Int4(param_path + "/lm_head/", 1, vocab_size, h_dim);;
}
