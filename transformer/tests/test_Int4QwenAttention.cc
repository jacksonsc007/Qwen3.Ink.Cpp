#include "Int4QwenAttention.h"
#include "utils.h"
#include "utils_memalloc.h"
#include <vector>
#include <sstream>

template <typename T>
void init_matrix(Matrix3D<T> & matrix, T value)
{
    for (int i = 0; i < matrix.length(); i++)
    {
        matrix.m_data[i] = value;
    }    
}

void random_init_matrix(Matrix3D<float> & matrix)
{
    for (int i = 0; i < matrix.length(); i++)
    {
        matrix.m_data[i] = (float) rand() / (float) RAND_MAX;
    }    
}


// test if module is grammatically correct
void test_model_pass()
{
    int bs = 1;
    int num_heads = 32;
    int num_layers = 32;
    int max_sqlen = 512;
    int embed_dim = 4096;
    int hidden_dim = 11008;
    int vocsize = 151936;
    int padding_idx = 1;
    int qk;  // group size
    // initialize
    qwen_config qwen_config(bs, num_heads, num_layers, max_sqlen, embed_dim, hidden_dim, vocsize, padding_idx);
    std::string param_path = "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/layer0/self_attn";
    Int4QwenAttention qwen_attention_layer(param_path, qwen_config);
    Int4QwenAttention::initialize_memory(qwen_config);

    // prepare input
    MemoryAllocator mem_buf;
    int sqlen = 10;
    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
    init_matrix<float>(hidden_states, 0.2f);
    // read_to_array("assets/llama/tests/atten/sqlen9/hidden_states.bin", hidden_states.m_data, b * sqlen * embed_dim);
    print_first_k_elelment("hidden_states", hidden_states.m_data, 10);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    init_matrix<float>(hidden_states, 0.3f);
    // read_to_array("assets/llama/tests/atten/sqlen9/attention_mask.bin", attention_mask.m_data, attention_mask.length());
    print_first_k_elelment("attention_mask", hidden_states.m_data, 10);
    printf("\e[31m[INFO]\e[m start testing...\n");
    Int4QwenAttention_input qwen_attention_input(hidden_states, attention_mask, 0);
    
    // evaluate
    Int4QwenAttention_output qwen_attention_output = qwen_attention_layer.forward(qwen_attention_input);
    printf("\e[31m[INFO]\e[m evaluation finished. Greate job!\n");
}

void prepare_decoder_attention_mask(Matrix3D<float> & attn_mask, int length, int past_length)
{
    int cur_sq_len = length - past_length;
    assert (cur_sq_len > 0);

    // Matrix3D<float> attn_mask(this->attention_mask_buf, 1, cur_sq_len, length);
    float min = std::numeric_limits<float>::lowest();
    for ( int i = 0; i < cur_sq_len; i++ )
        for ( int j = 0; j < length; j++ )
        {
            // NOTE <= here
            if (j <= (i + past_length))
                attn_mask(0, i, j) = 0.0;
            else
                attn_mask(0, i, j) = min;
        }
    
} 

void print_matrix(const Matrix3D<float> & mat)
{
    int x = mat.m_dim_x;
    int y = mat.m_dim_y;
    int z = mat.m_dim_z;
    for (int i = 0; i < x; i++)
    {
        std::string seperator(50, '=');
        printf("%s\n", seperator.c_str());
        for (int j = 0; j < y; j++)
        {
            for (int k = 0; k < z; k++)
            {
                printf("%8.3f ", mat(i, j, k));
            }
            printf("\n");
        }
        printf("\n");
    }    
}

// test if module is functinally correct
void test_model_correctness()
{
    int bs = 1;
    int num_heads = 32;
    int num_layers = 32;
    int max_sqlen = 512;
    int embed_dim = 4096;
    int hidden_dim = 11008;
    int vocsize = 151936;
    int padding_idx = 1;
    int qk;  // group size
    assert (bs == 1); // only support bs = 1
    // =================================
    // initialize
    // =================================
    qwen_config qwen_config(bs, num_heads, num_layers, max_sqlen, embed_dim, hidden_dim, vocsize, padding_idx);
    std::string int4_param_path = "../../INT4/models/qwen-7b-chat/transformer/layer0/self_attn";
    std::string param_path = "../../qwen-7b-chat/transformer/layer0/self_attn";
    Int4QwenAttention qwen_attention_layer(int4_param_path, qwen_config);
    Int4QwenAttention::initialize_memory(qwen_config);

    // =================================
    // prepare input
    // =================================
    MemoryAllocator mem_buf;
    // NOTE: this number should match the ground truth actication
    int sqlen = 20;
    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim); // (1, 20, 4096)
    // init_matrix<float>(hidden_states, 0.2f);
    // random_init_matrix(hidden_states);

    std::string input_attn_path = param_path + "/activation/input_attn.bin";
    read_to_array(input_attn_path.c_str(), hidden_states.m_data, bs * sqlen * embed_dim);
    print_first_k_elelment("hidden_states", hidden_states.m_data, 10);

    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    prepare_decoder_attention_mask(attention_mask, sqlen, 0); // prefill stage
    // init_matrix<float>(hidden_states, 0.3f);
    // read_to_array("assets/llama/tests/atten/sqlen9/attention_mask.bin", attention_mask.m_data, attention_mask.length());
    // print_first_k_elelment("attention_mask", hidden_states.m_data, 10);
    // print_matrix(attention_mask);

    printf("\e[31m[INFO]\e[m start testing...\n");
    Int4QwenAttention_input qwen_attention_input(hidden_states, attention_mask, 0);
    
    // =================================
    // evaluate
    // =================================
    Int4QwenAttention_output qwen_attention_output = qwen_attention_layer.forward(qwen_attention_input);

    Matrix3D<float> output_gt(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
    std::string output_attn_path = param_path + "/activation/output_attn.bin";
    read_to_array(output_attn_path.c_str(), output_gt.m_data, bs * sqlen * embed_dim);
    // print_first_k_elelment("output_attn_gt", output_gt.m_data, 10);
    // print_first_k_elelment("output_attn", qwen_attention_output.attn_output.m_data, 10);
    bool success = 0;
    success = check_two_equal(output_gt.m_data, qwen_attention_output.attn_output.m_data, output_gt.length());
    if (success)
        printf("\e[32m[Congrats]\e[m Attention test passed~\n");
    // =================================
    // check output activations if the outputs do not match
    // =================================
    if (!success)
    {
        std::vector<std::string> activatoin_names = {
            "input.bin",
            "q.bin",
            "k.bin",
            "v.bin",
            "q_with_rope.bin",
            "k_with_rope.bin",
            "final_q.bin",
            "final_k.bin",
            "final_v.bin",
            "attn_before_out.bin",
        };
        for (auto name : activatoin_names)
        {
            std::ostringstream ss;
            ss << param_path << "/activation/" << name;
            // output from cpp
            Matrix3D<float> out_cpp(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
            output_attn_path = ss.str();
            read_to_array(output_attn_path.c_str(), out_cpp.m_data,bs * sqlen * embed_dim);
            // output from python
            ss.str("");
            ss << param_path << "/activation/py_" << name;
            Matrix3D<float> out_py(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
            output_attn_path = ss.str();
            read_to_array(output_attn_path.c_str(), out_py.m_data,bs * sqlen * embed_dim);
            success = check_two_equal(out_cpp.m_data, out_py.m_data, out_cpp.length());
            if (success)
                printf("\e[31m[INFO]\e[m %s evaluation succeeds. Greate job!\n\n", name.c_str());
            else
                printf("\e[32m[INFO]\e[m %s Check more, but greate job!\n\n", name.c_str());
        }
    }


}

int main()
{
    // test_model_pass();
    test_model_correctness();
}


