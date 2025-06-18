#ifndef QWENFORCAUSALLM_H
#define QWENFORCAUSALLM_H

#include <memory>
#include "QwenDecoder.h"
#include "QwenOperator.h"
#include "common.h"
#include "operators.h"

struct Qwen3ForCausalLM_Input {
    Matrix3D<int> * input_ids;

    Qwen3ForCausalLM_Input(
        Matrix3D<int> * input_ids_
    ) : input_ids(input_ids_) {}
};

struct Qwen3ForCausalLM_Output {
    Matrix3D<float> logits;
    Qwen3ForCausalLM_Output(Matrix3D<float> && logits):logits(std::move(logits)){}
    Qwen3ForCausalLM_Output(const Qwen3ForCausalLM_Output &) = delete;
    Qwen3ForCausalLM_Output & operator=(const Qwen3ForCausalLM_Output &) = delete;

    Qwen3ForCausalLM_Output(Qwen3ForCausalLM_Output &&) = default;
    Qwen3ForCausalLM_Output & operator=(Qwen3ForCausalLM_Output &&) = default;
};


class Qwen3ForCausalLM {
   public:
    Qwen3ForCausalLM(std::string param_path, const struct qwen3_config config);
    // ~Fp32Qwen3ForCausalLM();

    struct Qwen3ForCausalLM_Output forward(const struct Qwen3ForCausalLM_Input& input);

   private:
    int past_sqlen;
    ModelContext context_;
    Qwen3Model model;
    Qwen_Linear_with_bias_Int4 lm_head;
    std::string profile_name = "Qwen3ForCausalLM";
    // Matrix3D<float> lm_head_output;
    Matrix3D<float> lm_head_weight;
};

#endif