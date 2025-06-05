#ifndef QWENFORCAUSALLM_H
#define QWENFORCAUSALLM_H

#include <memory>
#include "QwenDecoder.h"
#include "QwenOperator.h"

struct Qwen3ForCausalLM_Input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Qwen3ForCausalLM_Input() {}
    Qwen3ForCausalLM_Input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Qwen3ForCausalLM_Input(
        Matrix3D<int> input_ids_,
        std::vector<Matrix3D<float>> past_keys_, 
        std::vector<Matrix3D<float>> past_values_
    )
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) 
    {
        has_past_keys_values = true;
    }
};

struct Qwen3ForCausalLM_Output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<float>> past_keys, past_values;
};

class Qwen3ForCausalLM {
   public:
    Qwen3ForCausalLM(std::string param_path, const struct qwen3_config config);
    // ~Fp32Qwen3ForCausalLM();

    struct Qwen3ForCausalLM_Output forward(const struct Qwen3ForCausalLM_Input& input);

   private:
    Qwen3Model model;
    Qwen_Linear_with_bias_Int4 lm_head;
    std::string profile_name = "Qwen3ForCausalLM";
    std::shared_ptr<float> lm_head_output;
    std::shared_ptr<float> lm_head_weight;
};

#endif