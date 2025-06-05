#include "Fp32QwenDecoder.h"
#include "common.h"
#include "utils.h"
#include <memory>
#include <sstream>

void Fp32Qwen3Model::prepare_decoder_attention_mask(int length, int past_length, Matrix3D<float>& attn_mask)
{
    int cur_sq_len = length - past_length;
    assert (cur_sq_len > 0);

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

Fp32Qwen3Model::Fp32Qwen3Model(std::string param_path, const struct qwen3_config config){
    // allocate_aligned_memory(attention_mask_buf, sizeof(float) * config.max_sqlen * config.max_sqlen);
    // allocate_aligned_memory(last_hidden_states_buf, sizeof(float) * config.max_sqlen * config.embed_dim);
    voc_size = config.vocsize;
    hidden_dim = config.hidden_dim;
    max_sqlen = config.max_sqlen;
    num_layers = config.num_layers;
    bs = config.batchsize;
    this->param_path = param_path;
    
    // allocate space for intermediate values
    hidden_embed_ptr = std::shared_ptr<float> (
        new float[bs * max_sqlen * hidden_dim],
        std::default_delete<float []>()
    );
    attention_mask_buf = std::shared_ptr<float> (
        new float[bs * max_sqlen * max_sqlen],
        std::default_delete<float []>()
    );

    // work token embeddings
    wte_weight_ptr = std::shared_ptr<float> (
        new float[voc_size * hidden_dim],
        std::default_delete<float []>()
    );
    Matrix3D<float> wte_weight(wte_weight_ptr.get(), 1, voc_size, hidden_dim);
    wte = Embedding(hidden_dim, voc_size, wte_weight);
    printf("\e[31m[INFO]\e[m Loading weights for word token embeddings ...\n");
    load_Embedding_params(wte, param_path + "/embed_tokens");

    // final output norm
    output_norm_ptr = std::shared_ptr<float> (
        new float[bs * max_sqlen * hidden_dim],
        std::default_delete<float []>()
    );
    Matrix3D<float> output_norm_weight(output_norm_ptr.get(), 1, 1, hidden_dim); 
    printf("\e[31m[INFO]\e[m Loading weights for output norm ...\n");
    output_norm_weight.load((param_path + "/norm/weight.bin").c_str());
    output_norm = Qwen3RMSNorm(output_norm_weight);

    // decoder layers
    printf("\e[31m[INFO]\e[m Loading weights for Qwen Blocks ...\n");
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        IF_DEBUG(
            printf("\e[31m[INFO]\e[m Loading Qwen Block %d...\n", layer_idx);
        );
        std::string path = param_path + "/layers/layer" + std::to_string(layer_idx);
        Fp32Qwen3DecoderLayer layer = Fp32Qwen3DecoderLayer(path, config, layer_idx);
        layers.push_back(layer);
    }
}

Fp32Qwen3Model_Output Fp32Qwen3Model::forward(const struct Fp32Qwen3Model_Input &input) {
    PROFILE_START(this->profile_name);
    /*
    cur_sqlen should be 1 at autoregressive generation stage;
    past_sqlen records the total number of processed token length.
    */
    int cur_sqlen = input.input_ids.m_dim_z;
    int past_sqlen = 0;

    // ---
    // 1st stage: transform input tokens into embeddings
    // ---
    Matrix3D<float> token_embedding(hidden_embed_ptr.get(), bs, cur_sqlen, hidden_dim);
    wte.forward(input.input_ids, token_embedding);
    if (input.has_past_keys_values)
    {
        past_sqlen = input.past_keys[0].m_dim_y;
    }
    IF_DEBUG_IO(
        // std::string save_path = "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/activation/input_embeds.bin";
        std::string save_path = param_path + "/activation/token_embedding.bin";
        write_array_to_file(save_path.c_str(), token_embedding.m_data, token_embedding.length());
    );
    IF_DEBUG_DECODER(
        token_embedding.statistics();
    );

    // ---
    // 2nd stage: prepare causal attention mask
    // ---
    int entire_sq_len = cur_sqlen + past_sqlen;
    Matrix3D<float> attn_mask(this->attention_mask_buf.get(), bs, cur_sqlen, entire_sq_len);
    prepare_decoder_attention_mask(entire_sq_len, past_sqlen, attn_mask);

    // ---
    // 3rd stage: layer-by-layer inference
    // ---
    Matrix3D<float> hidden_states = token_embedding; 
    std::vector<Matrix3D<float>> past_keys, past_values;
    for (int i = 0; i < num_layers; i++)
    {
        IF_DEBUG(
            printf("\e[31m[INFO]\e[m evaluating layer %d ... \n", i);
        );
        if (!input.has_past_keys_values)
        {
            struct Fp32Qwen3DecoderLayer_Input  layer_input   = {hidden_states, attn_mask};
            struct Fp32Qwen3DecoderLayer_Output layer_output = this->layers[i].forward(layer_input);
            hidden_states = layer_output.hidden_states;
            past_keys.push_back(layer_output.past_key_value.first);
            past_values.push_back(layer_output.past_key_value.second);
        }
        else
        {
            struct Fp32Qwen3DecoderLayer_Input  layer_input = {hidden_states, attn_mask, input.past_keys[i], input.past_values[i]};
            struct Fp32Qwen3DecoderLayer_Output layer_output = this->layers[i].forward(layer_input);
            hidden_states = layer_output.hidden_states;
            past_keys.push_back(layer_output.past_key_value.first);
            past_values.push_back(layer_output.past_key_value.second);
        }
        
        IF_DEBUG_IO(
             std::ostringstream oss;
             std::string save_path;
             // input
            //  oss << "/root/workspace/tinyml/TinyChatEngine/qwen-7b-chat/transformer/activation/output_layer" <<
            //  i << ".bin";
             oss << param_path << "/activation/layer" << i << "/output.bin";
             save_path = oss.str();
             write_array_to_file(save_path.c_str(), hidden_states.m_data, hidden_states.length());
        );

        IF_DEBUG_DECODER(
            printf("\e[31m[INFO]\e[m layer %d output: ", i);
            hidden_states.statistics();
        );
    }
    // ---
    // 4th stage: output layernorm
    // ---
    Matrix3D<float> last_hidden_states(hidden_embed_ptr.get(), 1, cur_sqlen, hidden_dim);
    this->output_norm.forward(hidden_states, last_hidden_states);

    IF_DEBUG_DECODER(
        printf("\e[31m[INFO]\e[m decoder output: ");
        last_hidden_states.statistics();
    );

    struct Fp32Qwen3Model_Output output  = {last_hidden_states, past_keys, past_values}; 
    PROFILE_END(this->profile_name);
    return output;
}