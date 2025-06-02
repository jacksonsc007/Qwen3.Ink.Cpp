#include "Int4QwenDecoder.h"
#include "utils.h"
#include <memory>
#include <sstream>

Matrix3D<float> Int4QwenModel::prepare_decoder_attention_mask(int length, int past_length)
{
    int cur_sq_len = length - past_length;
    assert (cur_sq_len > 0);

    Matrix3D<float> attn_mask(this->attention_mask_buf, 1, cur_sq_len, length);
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
    return attn_mask;
    
} 

Int4QwenModel::Int4QwenModel(std::string param_path, const struct qwen_config config){
    allocate_aligned_memory(attention_mask_buf, sizeof(float) * config.max_sqlen * config.max_sqlen);
    allocate_aligned_memory(last_hidden_states_buf, sizeof(float) * config.max_sqlen * config.embed_dim);
    this->voc_size = config.vocsize;
    this->embed_dim = config.embed_dim;
    this->hidden_dim = config.hidden_dim;
    this->num_heads = config.num_heads;
    this->padding_idx = config.padding_idx;
    // wte TODO: why use new instead of allocate_aligned_memory?
    Matrix3D<float> wte_weight(new float[voc_size * embed_dim], 1, voc_size, embed_dim);
    this->wte = Embedding(embed_dim, voc_size, padding_idx, wte_weight);
    printf("\e[31m[INFO]\e[m Loading weights for wte ...\n");
    load_Embedding_params(this->wte, param_path + "/wte");
    // Norm
    Matrix3D<float> ln_F_weight(new float[embed_dim], 1, 1, embed_dim); 
    ln_F_weight.load((param_path + "/norm/weight.bin").c_str());
    printf("\e[31m[INFO]\e[m Loading weights for ln_f ...\n");
    this->ln_F = QwenInt4RMSNorm(ln_F_weight);
    // decoder layers
    printf("\e[31m[INFO]\e[m Loading weights for Qwen Blocks ...\n");
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        DEBUG_INS(std::cout << "Start loading layer:" << layer_idx << "..." << std::endl;)
        std::string path = param_path + "/layer" + std::to_string(layer_idx);
        Int4QwenBlock layer = Int4QwenBlock(path, config, layer_idx);
        this->layers.push_back(layer);
    }
}

Int4QwenModel_output Int4QwenModel::forward(const struct Int4QwenModel_input &input) {
    PROFILE_START(this->profile_name);
    int sqlen = input.input_ids.m_dim_z; // TODO: dim_z?
    int batch_size = input.input_ids.m_dim_x;
    int past_key_values_length = 0;
    // -----------------------------
    // 1st stage: transform input tokens into embeddings
    // -----------------------------
    float inputs_embeds_buf[sqlen * this->embed_dim];
    // std::unique_ptr<float []>  inputs_embeds_buf = std::make_unique<float []>(sqlen * this->embed_dim);
    Matrix3D<float> inputs_embeds(inputs_embeds_buf, 1, sqlen, this->embed_dim);
    this->wte.forward(input.input_ids, inputs_embeds);
    if (input.has_past_keys_values)
    {
        past_key_values_length = input.past_keys[0].m_dim_y;
    }
    
    // std::string save_path = "/root/workspace/tinyml/TinyChatEngine/llm/INT4/qwen-7b-chat/transformer/activation/input_embeds.bin";
    // write_array_to_file(save_path.c_str(), inputs_embeds.m_data, inputs_embeds.length());


    // -----------------------------
    // 2nd stage: prepare attention mask
    // -----------------------------
    Matrix3D<float> causal_attn_mask = this->prepare_decoder_attention_mask(sqlen + past_key_values_length, past_key_values_length);
    // -----------------------------
    // 3rd stage: layer-by-layer inference
    // -----------------------------
    // TODO: = is not overloaded, might be problems?
    Matrix3D<float> hidden_states = inputs_embeds; 
    std::vector<Matrix3D<float>> past_keys, past_values;
    for (int i = 0; i < this->layers.size(); i++)
    {

        #if DEBUG == 1
        printf("\e[31m[INFO]\e[m evaluating layer %d ... \n", i);
        #endif
        if (!input.has_past_keys_values)
        {
            struct Int4QwenBlock_input layer_input = {hidden_states, causal_attn_mask};
            struct Int4QwenBlock_output layer_output = this->layers[i].forward(layer_input);
            hidden_states = layer_output.hidden_states;
            past_keys.push_back(layer_output.past_key_value.first);
            past_values.push_back(layer_output.past_key_value.second);
        }
        else
        {
            struct Int4QwenBlock_input layer_input = {hidden_states, causal_attn_mask, input.past_keys[i], input.past_values[i]};
            struct Int4QwenBlock_output layer_output = this->layers[i].forward(layer_input);
            hidden_states = layer_output.hidden_states;
            past_keys.push_back(layer_output.past_key_value.first);
            past_values.push_back(layer_output.past_key_value.second);
        }
        
        // =================
        // Debug
        // =================
        // std::ostringstream oss;
        // std::string save_path;
        // // input
        // oss << "/root/workspace/tinyml/TinyChatEngine/llm/INT4/qwen-7b-chat/transformer/activation/output_layer" <<
        //     i << ".bin";
        // save_path = oss.str();
        // write_array_to_file(save_path.c_str(), hidden_states.m_data, hidden_states.length());
    }
    // -----------------------------
    // 4th stage: output layernorm
    // -----------------------------
    Matrix3D<float> last_hidden_states(last_hidden_states_buf, 1, sqlen, this->embed_dim);
    this->ln_F.forward(hidden_states, last_hidden_states);
    // -----------------------------
    // 5th stage: wrapping up
    // -----------------------------
    struct Int4QwenModel_output output  = {last_hidden_states, past_keys, past_values}; 
    PROFILE_END(this->profile_name);
    return output;
}