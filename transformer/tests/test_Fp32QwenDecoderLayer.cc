#include "Fp32QwenDecoderLayer.h"
#include "utils.h"
#include "utils_memalloc.h"

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

/*
@brief Feed each decoder layer with the gt input, compare the output with the ground truth output
*/
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
    // initialize
    qwen_config config(bs, num_heads, num_layers, max_sqlen, embed_dim, hidden_dim, vocsize, padding_idx);
    std::string fp32_param_path = "../qwen-7b-chat/transformer/";
    std::string param_path = "../qwen-7b-chat/transformer/";

    // prepare input
    MemoryAllocator mem_buf;
    // NOTE: this number should match the ground truth activation
    int sqlen = 20;
    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim); // (1, 20, 4096)

    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    prepare_decoder_attention_mask(attention_mask, sqlen, 0); // prefill stage
    // init_matrix<float>(hidden_states, 0.3f);
    // read_to_array("assets/llama/tests/atten/sqlen9/attention_mask.bin", attention_mask.m_data, attention_mask.length());
    // print_first_k_elelment("attention_mask", hidden_states.m_data, 10);
    // print_matrix(attention_mask);
    Matrix3D<float> output_gt(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);

    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
    {
        
    std::ostringstream ss;
    ss << param_path << "layer" << layer_idx << "/activation/input_gt.bin";
    std::string input_attn_path = ss.str();
    read_to_array(input_attn_path.c_str(), hidden_states.m_data, bs * sqlen * embed_dim);
    // print_first_k_elelment("hidden_states", hidden_states.m_data, 10);

    printf("\e[31m[INFO]\e[m start testing QwenBlock %d...\n", layer_idx);
    Fp32QwenBlock_input input(hidden_states, attention_mask);
    // evaluate
    ss.str("");
    ss << fp32_param_path << "layer" << layer_idx << "/";
    std::string model_path = ss.str();
    Fp32QwenBlock layer(model_path, config, layer_idx);
    Fp32QwenBlock_output output = layer.forward(input);
    
    // compare with gt
    ss.str("");
    ss << param_path << "layer" << layer_idx << "/activation/output_gt.bin";
    std::string output_attn_path = ss.str();
    read_to_array(output_attn_path.c_str(), output_gt.m_data, bs * sqlen * embed_dim);
    print_first_k_elelment("output_gt", output_gt.m_data, 20);
    print_first_k_elelment("output   ", output.hidden_states.m_data, 20);

    bool success = check_two_equal(output_gt.m_data, output.hidden_states.m_data, output_gt.length());
    if (success)
        printf("\e[32m[Congrats]\e[m evaluation finished. Greate job!\n");
    else
        printf("\e[31m[INFO]\e[m Check more, but greate job!\n");
    }

}

int main()
{
    test_model_correctness();
}


