#include "Fp32QwenForCausalLM.h"


struct Fp32QwenForCausalLM_output Fp32QwenForCausalLM::forward(const struct Fp32QwenForCausalLM_input& input)
{
    PROFILE_START(profile_name);
    // -----------------------------
    // 1st stage: setting paras and buffers
    // -----------------------------
    int sqlen = input.input_ids.m_dim_z; // TODO: dim_z?
    struct Fp32QwenModel_output decoder_output;
    // -----------------------------
    // 2nd stage: evaluate decoder
    // -----------------------------
    if (input.has_past_keys_values)
    {
        struct Fp32QwenModel_input decoder_input = {input.input_ids, input.past_keys, input.past_values};
        decoder_output = this->decoder.forward(decoder_input);
    }
    else
    {
        struct Fp32QwenModel_input decoder_input = {input.input_ids};
        decoder_output = this->decoder.forward(decoder_input);
    }

    // -----------------------------
    // 3rd stage: evaluate head to get logits
    // TODO: We may no need to compute the logits of whole sequence, just the last one
    // -----------------------------
    Matrix3D<float> logits(logits_output, 1, sqlen, this->decoder.voc_size);
    this->lm_head.forward(decoder_output.last_hidden_state, logits);
    // -----------------------------
    // 4th stage: wrapping up
    // -----------------------------
    Fp32QwenForCausalLM_output output = {logits, decoder_output.past_keys, decoder_output.past_values};

    // -----------------------------
    // Debug
    // -----------------------------
    // std::string save_path = "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/activation/input_lm_head.bin";
    // write_array_to_file(save_path.c_str(), decoder_output.last_hidden_state.m_data, decoder_output.last_hidden_state.length());
    // save_path = "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/activation/output_lm_head.bin";
    // write_array_to_file(save_path.c_str(), logits.m_data, logits.length());
    // exit(1);
    PROFILE_END(profile_name);
    return output;
}

Fp32QwenForCausalLM::Fp32QwenForCausalLM(std::string param_path, const struct qwen_config config)
{
   allocate_aligned_memory(logits_output, config.max_sqlen * config.vocsize * sizeof(float));
   allocate_aligned_memory(lm_head_weight, config.embed_dim * config.vocsize * sizeof(float));

   this->decoder = Fp32QwenModel(param_path + "/transformer", config);
   this->lm_head = 
        Linear_FP(Matrix3D<float>(lm_head_weight, 1, config.vocsize, config.embed_dim), param_path + "/lm_head.bin");
}

Fp32QwenForCausalLM::~Fp32QwenForCausalLM()
{
    printf("\e[31m[Error]\e[m Deallocator of Fp32QwenForCausalLM is not implemented yet!");
}