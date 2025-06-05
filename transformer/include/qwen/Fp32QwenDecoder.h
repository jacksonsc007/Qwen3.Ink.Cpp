#ifndef _Fp32QwenDecoder_h
#define _Fp32QwenDecoder_h

#include <memory>
#include <vector>

#include "Fp32QwenDecoderLayer.h"
#include "common.h"
#include "operators.h"
#include "Fp32QwenOperator.h"

struct Fp32Qwen3Model_Output {
    Matrix3D<float> last_hidden_state;
    std::vector<Matrix3D<float>> past_keys, past_values;
};

struct Fp32Qwen3Model_Input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Fp32Qwen3Model_Input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Fp32Qwen3Model_Input(
        Matrix3D<int> input_ids_,
        std::vector<Matrix3D<float>> past_keys_,
        std::vector<Matrix3D<float>> past_values_
    ): input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};


class Fp32Qwen3Model {
public:
    // member
    int voc_size, hidden_dim, num_heads, num_layers, max_sqlen, bs;
    Embedding wte;
    Qwen3RMSNorm output_norm;
    std::vector<Fp32Qwen3DecoderLayer> layers;
    std::string profile_name = "Fp32Qwen3Model";
    float* last_hidden_states_buf;
    std::shared_ptr<float> wte_weight_ptr;
    std::shared_ptr<float> output_norm_ptr;
    std::shared_ptr<float> hidden_embed_ptr;
    std::shared_ptr<float> attention_mask_buf;
    // the directory of weights
    std::string param_path;

    // method
    Fp32Qwen3Model() = default;
    Fp32Qwen3Model(std::string param_path, const struct qwen3_config config);
    Fp32Qwen3Model_Output forward(const struct Fp32Qwen3Model_Input &input);
    void prepare_decoder_attention_mask(int length, int past_length, Matrix3D<float> &);




};

#endif