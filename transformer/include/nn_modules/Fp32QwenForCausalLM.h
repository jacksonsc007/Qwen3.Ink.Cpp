#ifndef FP32QWENFORCAUSALLM_H
#define FP32QWENFORCAUSALLM_H

#include "Fp32QwenDecoder.h"

struct Fp32QwenForCausalLM_output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Fp32QwenForCausalLM_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Fp32QwenForCausalLM_input() {}
    Fp32QwenForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Fp32QwenForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                               std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Fp32QwenForCausalLM {
   public:
    Fp32QwenForCausalLM(std::string param_path, const struct qwen_config config);
    ~Fp32QwenForCausalLM();

    struct Fp32QwenForCausalLM_output forward(const struct Fp32QwenForCausalLM_input& input);

   private:
    Fp32QwenModel decoder;
    Linear_FP lm_head;
    std::string profile_name = "Fp32QwenForCausalLM";
    float* logits_output;
    float* lm_head_weight;
};

#endif