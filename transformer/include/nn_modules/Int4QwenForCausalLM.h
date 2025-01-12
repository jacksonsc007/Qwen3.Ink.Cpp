#ifndef FP32QWENFORCAUSALLM_H
#define FP32QWENFORCAUSALLM_H

#include "Int4QwenDecoder.h"

struct Int4QwenForCausalLM_output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Int4QwenForCausalLM_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Int4QwenForCausalLM_input() {}
    Int4QwenForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Int4QwenForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                               std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Int4QwenForCausalLM {
   public:
    Int4QwenForCausalLM(std::string param_path, const struct qwen_config config);
    ~Int4QwenForCausalLM();

    struct Int4QwenForCausalLM_output forward(const struct Int4QwenForCausalLM_input& input);

   private:
    Int4QwenModel decoder;
    Qwen_Linear_with_bias_Int4 lm_head;
    std::string profile_name = "Int4QwenForCausalLM";
    float* logits_output;
    float* lm_head_weight;
};

#endif