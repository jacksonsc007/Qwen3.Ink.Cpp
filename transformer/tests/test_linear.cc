#include <cmath>

#include "common.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"
#include "QwenOperator.h"
// #include <sstream>

void test_linear() {
    std::string param_path = "model_weights/int4/qwen3-8b-A81W41/model/layers/layer0/self_attn";
    // std::string param_path = "model_weights/int4/qwen3_8b_A80W40/model/layers/layer0/self_attn";
    int q_dim = 4096;
    int hidden_dim = 4096;
    int sqlen = 10;
    ModelContext ctx;
    size_t max_weight_rows = 4096;
    size_t max_weight_cols = 151936;
    size_t num_q_blocks = max_weight_rows * max_weight_cols / QK;
    size_t size_per_q_block = (
        sizeof(float) * 4 + // scaling factors, min, zero, scaled sum
        (sizeof(uint8_t) / 2) * QK + // quantized weights
        (sizeof(uint8_t) / 2) * QK // prepare for mixture of A80W40 and A81W41
    );
    size_t weight_buffer_size = (
        num_q_blocks * size_per_q_block
    );
    ctx.repack_buffer = std::make_unique<int8_t []>(weight_buffer_size); // 2 int4 weights in one byte

    size_t max_activation_cols = 12288; // 12288 is the maximum hidden dimension
    size_t max_activation_rows = 4096;
    num_q_blocks = max_activation_rows * max_activation_cols / QK;
    size_per_q_block = (
        sizeof(float) * 4 + // scaling factors, min, zero, scaled sum
        (sizeof(uint8_t)) * QK 
    );

    size_t activation_buffer_size = (
        num_q_blocks * size_per_q_block
    );
    ctx.activation_buffer = std::make_unique<int8_t []>(activation_buffer_size); // 2 int4 weights in one byte

    
    //
    Qwen_Linear_with_bias_Int4 q_proj(&ctx, param_path + "/q_proj/", 1, q_dim, hidden_dim);
    
    Matrix3D<float> input(1, sqlen, hidden_dim);
    // Matrix3D<float> output_gt(1, sqlen, hidden_dim);
    std::string input_path = "tests/assets/qwen3_attention_input.bin";
    std::string output_gt_path = "tests/assets/qwen3_attention_q_unshape.bin";
    input.load(input_path.c_str());
    // output_gt.load(output_gt_path.c_str());
    
    Matrix3D<float> output_linear = q_proj.forward(input);
    output_linear.compare_with_gt(output_gt_path);
}


int main() {
    test_linear();
    Profiler::getInstance().report_internal();
}
