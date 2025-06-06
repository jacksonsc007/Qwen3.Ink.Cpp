#include "QwenDecoder.h"
#include "common.h"
#include "utils.h"

void Qwen3Model::prepare_decoder_attention_mask(int length, int past_length, Matrix3D<float>& attn_mask)
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

Qwen3Model::Qwen3Model(std::string param_path, const struct qwen3_config config){
    // allocate_aligned_memory(attention_mask_buf, sizeof(float) * config.max_sqlen * config.max_sqlen);
    // allocate_aligned_memory(last_hidden_states_buf, sizeof(float) * config.max_sqlen * config.embed_dim);
    voc_size = config.vocsize;
    hidden_dim = config.hidden_dim;
    max_sqlen = config.max_sqlen;
    num_layers = config.num_layers;
    bs = config.batchsize;
    this->param_path = param_path;

    IF_DEBUG(
        printf("\e[31m[INFO]\e[m Loading weights for word token embeddings ...\n");
    )
    wte = Embedding(hidden_dim, voc_size);
    wte.load(param_path + "/embed_tokens");

    // final output norm
    printf("\e[31m[INFO]\e[m Loading weights for output norm ...\n");
    output_norm = Qwen3RMSNorm(hidden_dim);
    output_norm.load((param_path + "/norm/weight.bin").c_str());

    // decoder layers
    printf("\e[31m[INFO]\e[m Loading weights for Qwen Blocks ...\n");
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        IF_DEBUG(
            printf("\e[31m[INFO]\e[m Loading Qwen Block %d...\n", layer_idx);
        );
        std::string path = param_path + "/layers/layer" + std::to_string(layer_idx);
        Qwen3DecoderLayer layer = Qwen3DecoderLayer(path, config, layer_idx);
        layers.push_back(layer);
    }
}

Qwen3Model_Output Qwen3Model::forward(const struct Qwen3Model_Input &input) {
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
    PROFILE_START(profile_name + "::embedding");
    Matrix3D<float> token_embedding  = wte.forward(input.input_ids);
    PROFILE_END(profile_name + "::embedding");
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
    PROFILE_START(profile_name + "::attention mask generation");
    int entire_sq_len = cur_sqlen + past_sqlen;
    Matrix3D<float> attn_mask(bs, cur_sqlen, entire_sq_len);
    prepare_decoder_attention_mask(entire_sq_len, past_sqlen, attn_mask);
    PROFILE_END(profile_name + "::attention mask generation");

    // ---
    // 3rd stage: layer-by-layer inference
    // ---
    PROFILE_START(profile_name + "::decoder layers");
    Matrix3D<float> hidden_states(std::move(token_embedding)); 
    std::vector<Matrix3D<float>> past_keys, past_values;
    for (int i = 0; i < num_layers; i++)
    {
        IF_DEBUG(
            printf("\e[31m[INFO]\e[m evaluating layer %d ... \n", i);
        );
        if (!input.has_past_keys_values)
        {
            struct Qwen3DecoderLayer_Input  layer_input   = {hidden_states, attn_mask};
            struct Qwen3DecoderLayer_Output layer_output = this->layers[i].forward(layer_input);
            hidden_states = layer_output.hidden_states;
            past_keys.push_back(layer_output.past_key_value.first);
            past_values.push_back(layer_output.past_key_value.second);
        }
        else
        {
            struct Qwen3DecoderLayer_Input  layer_input = {hidden_states, attn_mask, input.past_keys[i], input.past_values[i]};
            struct Qwen3DecoderLayer_Output layer_output = this->layers[i].forward(layer_input);
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
    PROFILE_END(profile_name + "::decoder layers");
    // ---
    // 4th stage: output layernorm
    // ---
    PROFILE_START(profile_name + "::output layernorm");
    Matrix3D<float> last_hidden_states = output_norm.forward(hidden_states);
    PROFILE_END(profile_name + "::output layernorm");

    IF_DEBUG_DECODER(
        printf("\e[31m[INFO]\e[m decoder output: ");
        last_hidden_states.statistics();
    );

    struct Qwen3Model_Output output  = {last_hidden_states, past_keys, past_values}; 
    PROFILE_END(this->profile_name);
    return output;
}