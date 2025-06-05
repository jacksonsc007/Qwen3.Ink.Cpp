#ifndef FP32QWENFORCAUSALLM_H
#define FP32QWENFORCAUSALLM_H

#include <memory>
#include "Fp32QwenDecoder.h"

struct Fp32Qwen3ForCausalLM_Input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Fp32Qwen3ForCausalLM_Input() {}
    Fp32Qwen3ForCausalLM_Input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Fp32Qwen3ForCausalLM_Input(
        Matrix3D<int> input_ids_,
        std::vector<Matrix3D<float>> past_keys_, 
        std::vector<Matrix3D<float>> past_values_
    )
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) 
    {
        has_past_keys_values = true;
    }
};

struct Fp32Qwen3ForCausalLM_Output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<float>> past_keys, past_values;
};

class Fp32Qwen3ForCausalLM {
   public:
    Fp32Qwen3ForCausalLM(std::string param_path, const struct qwen3_config config);
    // ~Fp32Qwen3ForCausalLM();

    struct Fp32Qwen3ForCausalLM_Output forward(const struct Fp32Qwen3ForCausalLM_Input& input);

   private:
    Fp32Qwen3Model model;
    LinearFp32 lm_head;
    std::string profile_name = "Fp32Qwen3ForCausalLM";
    std::shared_ptr<float> lm_head_output;
    std::shared_ptr<float> lm_head_weight;
};

#endif