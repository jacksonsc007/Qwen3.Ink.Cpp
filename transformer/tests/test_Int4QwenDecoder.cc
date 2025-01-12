#include "Int4QwenDecoder.h"
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
    qwen_config qwen_config(bs, num_heads, num_layers, max_sqlen, embed_dim, hidden_dim, vocsize, padding_idx);
    
    std::string int4_param_path = "../../INT4/models/qwen-7b-chat/transformer";
    std::string param_path = "../../qwen-7b-chat/transformer"; // the path that store the gt from python model.
    
    Int4QwenModel transformer(int4_param_path, qwen_config);

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
    Int4QwenModel_input input = Int4QwenModel_input(input_ids);

    // =======================
    // evaluate
    // =======================
    bool success = 0;
    printf("\e[31m[INFO]\e[m start testing Qwen transformer decoder ...\n");
    Int4QwenModel_output output = transformer.forward(input);

    Matrix3D<float> output_gt(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
    std::string output_gt_path = param_path + "/activation/output_gt.bin";
    read_to_array(output_gt_path.c_str(), output_gt.m_data, bs * sqlen * embed_dim);
    print_first_k_elelment("output_gt", output_gt.m_data, 20);
    print_first_k_elelment("output   ", output.last_hidden_state.m_data, 20);
    printf("\n");

    success = check_two_equal(output_gt.m_data, output.last_hidden_state.m_data, output_gt.length());
    if (success)
        printf("\e[32m[Congrats]\e[m Qwen transformer evaluation finished. Greate job!\n");
    else
    {
        printf("\e[31m[INFO]\e[m evaluation Failed. Checking activations: \n");
        // check input embeds
        std::string output_path = 
            int4_param_path + "/activation/input_embeds.bin";
        // output from cpp
        Matrix3D<float> out_int4_cpp(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
        read_to_array(output_path.c_str(), out_int4_cpp.m_data,bs * sqlen * embed_dim);
        // output from python
        output_path = 
            param_path + "/activation/input_embeds_gt.bin";
        Matrix3D<float> out_py(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
        read_to_array(output_path.c_str(), out_py.m_data,bs * sqlen * embed_dim);
        success = check_two_equal(out_int4_cpp.m_data, out_py.m_data, out_int4_cpp.length());
        if (success)
            printf("\e[32m[INFO]\e[m input_embedding evaluation succeeds. Greate job!\n\n");
        else
            printf("\e[31m[INFO]\e[m input_embedding Check more, but greate job!\n\n");
        
        for (int layer_idx = 0 ; layer_idx < 32; layer_idx++)
        {
            std::ostringstream ss;
            std::string output_path;
            // output from cpp
            ss << int4_param_path << "/activation/output_layer" << layer_idx << ".bin";
            Matrix3D<float> out_cpp(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
            output_path = ss.str();
            read_to_array(output_path.c_str(), out_cpp.m_data,bs * sqlen * embed_dim);

            // output from python
            ss.str("");
            ss << param_path << "/activation/output_layer" << layer_idx << "_gt.bin";
            Matrix3D<float> out_py(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
            output_path = ss.str();
            read_to_array(output_path.c_str(), out_py.m_data,bs * sqlen * embed_dim);
            success = check_two_equal(out_cpp.m_data, out_py.m_data, out_cpp.length());
            if (success)
                printf("\e[32m[INFO]\e[m %d evaluation succeeds. Greate job!\n\n", layer_idx);
            else
                printf("\e[31m[INFO]\e[m %d Check more, but greate job!\n\n", layer_idx);
            
            // output from fp32 cpp
            ss.str("");
            ss << param_path << "/activation/output_layer" << layer_idx << ".bin";
            Matrix3D<float> out_fp32(mem_buf.get_fpbuffer(embed_dim * sqlen), bs, sqlen, embed_dim);
            output_path = ss.str();
            read_to_array(output_path.c_str(), out_fp32.m_data,bs * sqlen * embed_dim);
            success = check_two_equal(out_cpp.m_data, out_fp32.m_data, out_cpp.length());
            if (success)
                printf("\e[32m[INFO]\e[m %d evaluation succeeds. Greate job!\n\n", layer_idx);
            else
                printf("\e[31m[INFO]\e[m %d Check more, but greate job!\n\n", layer_idx);
        }
    }

}

int main()
{
    test_model_correctness();
}


