#include "Fp32QwenForCausalLM.h"
#include "utils.h"
#include "utils_memalloc.h"
#include <filesystem>

template <typename T>
void init_matrix(Matrix3D<T> & matrix, T value)
{
    for (int i = 0; i < matrix.length(); i++)
    {
        matrix.m_data[i] = value;
    }    
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
    // =======================
    // initialize
    // =======================
    model_config qwen_config(bs, num_heads, num_layers, max_sqlen, embed_dim, hidden_dim, vocsize, padding_idx);
    std::string param_path = "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat";
    Fp32QwenForCausalLM model = Fp32QwenForCausalLM(param_path, qwen_config);

    // =======================
    // prepare input
    // =======================
    MemoryAllocator mem_buf;
    // NOTE: this number should match the ground truth actication
    int sqlen = 20;
    Matrix3D<int> input_ids(mem_buf.get_intbuffer(sqlen), bs, 1, sqlen); // (1, 1, 20)
    // load
    std::string input_path = param_path + "/activation/input_ids_gt.bin";
    input_ids.load(input_path.c_str());
    Fp32QwenForCausalLM_input input = {input_ids};

    // =======================
    // evaluate for 1st iteration without KV cache
    // =======================
    bool success = 0;
    printf("\e[31m[INFO]\e[m start testing QwenForCausalLM ...\n");
    Fp32QwenForCausalLM_output output = model.forward(input);

    Matrix3D<float> output_gt(mem_buf.get_fpbuffer(bs* vocsize * sqlen), bs, sqlen, vocsize);
    std::string output_gt_path = param_path + "/activation/output_gt.bin";
    read_to_array(output_gt_path.c_str(), output_gt.m_data, bs * sqlen * vocsize);
    print_first_k_elelment("output_gt", output_gt.m_data, 20);
    print_first_k_elelment("output   ", output.logits.m_data, 20);
    printf("\n");

    success = check_two_equal(output_gt.m_data, output.logits.m_data, output_gt.length());
    if (success)
        printf("\e[32m[Congrats]\e[m QwenForCausalLM evaluation finished. Greate job!\n");
    else
    {
        printf("\e[31m[INFO]\e[m QwenForCausalLM evaluation Failed. Checking activations: \n");
        
        
        // check input embeds
        std::string output_path = 
            "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/activation/last_hidden_states.bin";
        // output from cpp
        Matrix3D<float> out_cpp(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
        read_to_array(output_path.c_str(), out_cpp.m_data,bs * sqlen * embed_dim);
        // output from python
        output_path = 
            "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/activation/last_hidden_states_gt.bin";
        Matrix3D<float> out_py(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
        read_to_array(output_path.c_str(), out_py.m_data,bs * sqlen * embed_dim);
        success = check_two_equal(out_cpp.m_data, out_py.m_data, out_cpp.length());
        if (success)
            printf("\e[32m[Congrat]\e[m last_hidden_states evaluation succeeds. Greate job!\n\n");
        else
            printf("\e[31m[INFO]\e[m last_hidden_states Check more, but greate job!\n\n");
    }

    // =======================
    // evaluate for 2nd iteration with KV cache
    // =======================
    // =======================
    // prepare input
    // =======================
    Matrix3D<int> input_ids_2nd(mem_buf.get_intbuffer(1), bs, 1, 1); // each iteration generates 1 new token
    // load
    input_path = param_path + "/activation/input_ids_2nd_gt.bin";
    input_ids_2nd.load(input_path.c_str());
    Fp32QwenForCausalLM_input input_2nd = {input_ids_2nd, output.past_keys, output.past_values};

    success = 0;
    printf("\e[31m[INFO]\e[m start testing QwenForCausalLM with KV Cache...\n");
    Fp32QwenForCausalLM_output output_2nd = model.forward(input_2nd);

    Matrix3D<float> output_gt_2nd(mem_buf.get_fpbuffer(bs* vocsize * 1), bs, 1, vocsize);
    std::string output_gt_2nd_path = param_path + "/activation/output_2nd_gt.bin";
    read_to_array(output_gt_2nd_path.c_str(), output_gt_2nd.m_data, bs * sqlen * vocsize);
    print_first_k_elelment("output_gt_2nd", output_gt_2nd.m_data, 20);
    print_first_k_elelment("output_2nd  ", output_2nd.logits.m_data, 20);
    printf("\n");

    success = check_two_equal(output_gt_2nd.m_data, output_2nd.logits.m_data, output_gt_2nd.length());
    if (success)
        printf("\e[32m[Congrats]\e[m QwenForCausalLM with KV cache evaluation finished. Greate job!\n");
    else
    {
    }

}

int main()
{
    test_model_correctness();
}


