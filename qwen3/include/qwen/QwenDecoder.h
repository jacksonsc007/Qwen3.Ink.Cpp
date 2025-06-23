#ifndef _QwenDecoder_h
#define _QwenDecoder_h

#include <memory>
#include <vector>

#include "QwenDecoderLayer.h"
#include "common.h"
#include "operators.h"
#include "QwenOperator.h"

struct Qwen3Model_Output {
    Matrix3D<float> last_hidden_state;
    
    Qwen3Model_Output(Matrix3D<float> && last_hidden_state):last_hidden_state(std::move(last_hidden_state))
    {}
    Qwen3Model_Output(const Qwen3Model_Output &) = delete;
    Qwen3Model_Output & operator=(const Qwen3Model_Output &) = delete;

    Qwen3Model_Output(Qwen3Model_Output &&) = default;
    Qwen3Model_Output & operator=(Qwen3Model_Output &&) = default;

};

struct Qwen3Model_Input {
    Matrix3D<int> * input_ids;
    int past_sqlen;

    Qwen3Model_Input(
        Matrix3D<int>* input_ids_,
        int past_sqlen_
    ): input_ids(input_ids_), past_sqlen(past_sqlen_)
    {

    }
};


class Qwen3Model {
    ModelContext * context_;
public:
    // member
    int voc_size, hidden_dim, num_heads, num_layers, max_sqlen, bs;
    Qwen3RMSNorm output_norm;
    std::vector<Qwen3DecoderLayer> layers;
    std::string profile_name = "QwenDecoder";
    std::string forward_profile_name;
    float* last_hidden_states_buf;

    Embedding wte;

    // the directory of weights
    std::string param_path;

    // method
    Qwen3Model() = default;
    Qwen3Model(ModelContext * ctx, std::string param_path, const struct qwen3_config config);
    Qwen3Model_Output forward(const struct Qwen3Model_Input &input);
    void prepare_decoder_attention_mask(int length, int past_length, Matrix3D<float> &);




};

#endif