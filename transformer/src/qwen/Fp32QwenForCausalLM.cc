#include "Fp32QwenForCausalLM.h"

#include <memory>

#include "common.h"

struct Fp32Qwen3ForCausalLM_Output Fp32Qwen3ForCausalLM::forward(const struct Fp32Qwen3ForCausalLM_Input& input) {
    PROFILE_START(profile_name);
    // -----------------------------
    // 1st stage: setting paras and buffers
    // -----------------------------
    struct Fp32Qwen3Model_Output decoder_output;

    // -----------------------------
    // 2nd stage: evaluate decoder
    // -----------------------------
    if (input.has_past_keys_values) {
        // autoregressive generation stage
        struct Fp32Qwen3Model_Input decoder_input = {input.input_ids, input.past_keys, input.past_values};
        decoder_output = this->model.forward(decoder_input);
    } else {
        // prompt stage
        struct Fp32Qwen3Model_Input decoder_input = {input.input_ids};
        decoder_output = this->model.forward(decoder_input);
    }

    // -----------------------------
    // 3rd stage: evaluate head to get logits
    // -----------------------------
    int bs = decoder_output.last_hidden_state.m_dim_x;
    int sqlen = decoder_output.last_hidden_state.m_dim_y;
    int h_dim = decoder_output.last_hidden_state.m_dim_z;
    ASSERT(bs == 1);

    Matrix3D<float> logits(lm_head_output.get(), 1, sqlen, this->model.voc_size);
    float* last_token_last_h_ptr = &decoder_output.last_hidden_state(0, sqlen - 1, 0);
    Matrix3D<float> last_token_last_h(last_token_last_h_ptr, 1, sqlen, h_dim);
    this->lm_head.forward(last_token_last_h, logits);

    Fp32Qwen3ForCausalLM_Output output = {logits, decoder_output.past_keys, decoder_output.past_values};

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

Fp32Qwen3ForCausalLM::Fp32Qwen3ForCausalLM(std::string param_path, const struct qwen3_config config) {
    int bs = config.batchsize;
    int h_dim = config.hidden_dim;
    int max_sqlen = config.max_sqlen;
    int vocab_size = config.vocsize;
    lm_head_output = std::shared_ptr<float>(
        new float[bs * max_sqlen * vocab_size],
        std::default_delete<float[]>()
    );
    lm_head_weight = std::shared_ptr<float>(
        new float[h_dim * vocab_size],
        std::default_delete<float[]>()
    );

    this->model = Fp32Qwen3Model(param_path + "/model", config);

    /*
    The linear weights are serialized from PyTorch, which has shape (out_dim, in_dim)
    */
    this->lm_head =
        LinearFp32(Matrix3D<float>(lm_head_weight.get(), 1, vocab_size, h_dim), param_path + "/lm_head.bin");
}
