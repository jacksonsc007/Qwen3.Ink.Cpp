#include "common.h"
#include "utils.h"
#include "Generate.h"
#include "QwenTokenizer.h"
#include "QwenForCausalLM.h"

std::vector<int> QwenGenerate(void *model_ptr, std::string text,
    const struct qwen_params generation_config, std::string tiktoken_path, bool interactive, qwen3_config config)
{
    // TODO: Distinguish the notion of context here with the size of KV Cache
    const int32_t max_context_length = generation_config.n_ctx;
    std::vector<int> last_n_tokens(max_context_length); // hold the generated tokens as context
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    // printf("\e[31m[INFO]\e[m Context Length: (%zu / %d)\n", last_n_tokens.size(), max_context_length);
    std::vector<int> embd; 
    std::vector<int> generate_ids; // return value

    const int max_token_length = 512;
    const int vocab_size = generation_config.n_vocab;
    std::vector<int> input_ids(max_token_length);
    // ===========================
    // Stage1 :Tokenize the input texts
    // ===========================
    #if DEBUG==1
    printf("\e[31m[INFO]\e[m Building Tokenizer...\n");
    #endif
    QwenTokenizer tokenizer = QwenTokenizer(tiktoken_path, config);
    #if DEBUG==1
    printf("\e[32m[INFO]\e[m Tokenizer built successfully\n");
    #endif
    input_ids = tokenizer.encode(text, max_token_length);
    
    if (interactive) std::cout << "ASSISTANT: " << std::endl;

    bool has_past_kv = false;
    bool previous_two_hash = false;
    std::vector<Matrix3D<float>> past_keys, past_values; // Hold KV Cache
    int n_remain = generation_config.n_predict;
    int stop_generation_tolerance = 2;
    while (n_remain != 0 && stop_generation_tolerance)
    {
        std::vector<float> logits(vocab_size);
        
        int sqlen = 1;

    // ===========================
    // Stage2: Model evaluation
    // ===========================
        Qwen3ForCausalLM *model = static_cast<Qwen3ForCausalLM *>(model_ptr);
        Qwen3ForCausalLM_Input model_input;
        Qwen3ForCausalLM_Output model_output;
        
        if (has_past_kv)
        {
            PROFILE_START("[ P Stage ]");
            assert (sqlen == 1);
            Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
            model_input = {input_ids_mat, past_keys, past_values};
            model_output = model->forward(model_input);
            PROFILE_END("[ P Stage ]");
        }
        else
        {
            PROFILE_START("[ AR Stage ]");
            sqlen = input_ids.size();
            Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
            model_input = {input_ids_mat};
            model_output = model->forward(model_input);
            PROFILE_END("[ AR Stage ]");
        }
        past_keys = model_output.past_keys;
        past_values = model_output.past_values;
        // we only need the logit of last token
        Matrix3D<float> last_token_logits = model_output.logits;
        memcpy(logits.data(), last_token_logits.data(), vocab_size*sizeof(float));
        has_past_kv = true;
    // ===========================
    // Stage3: Sampling strategy
    // ===========================
        PROFILE_START("[ Sampling ]");
        std::vector<token_data> candidates; 
        candidates.reserve(vocab_size);
        for (int token_id = 0; token_id < vocab_size; token_id++)
        {
            candidates.emplace_back(token_data{token_id, logits[token_id], 0.0f});
        }
        token_data_array candidiate_p = {candidates.data(), candidates.size(), false};
        // step 1: Apply sampling preference: use new tokens or repeat tokens?
        // Please refer to https://docs.vllm.ai/en/stable/dev/sampling_params.html for more details
        const int32_t repeat_last_n = generation_config.repeat_last_n < 0 ? max_context_length : generation_config.repeat_last_n;
        // make sure the number of tokens to penalize do not exceed the current context length
        int32_t last_n_repeat = std::min(
            std::min( (int)last_n_tokens.size(), repeat_last_n),
            max_context_length
        );
        // apply repetition penalty
        sample_repetition_penalty(
            &candidiate_p,
            last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
            last_n_repeat,
            generation_config.repeat_penalty
        );
        // apply frequency and presence penalty
        sample_frequency_and_presence_penalties(
            &candidiate_p,
            last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
            last_n_repeat,
            generation_config.frequency_penalty,
            generation_config.presence_penalty
        );
    
        // step 2: sampling
        const float temperature = generation_config.temp;
        const int mirostat = generation_config.mirostat; // enable mirostat sample algorithm
        int next_token_id = 0;
        if (temperature <= 0)
        {
            // apply greedy search
            next_token_id = sample_token_greedy(&candidiate_p);
        }
        else
        {
            if (mirostat != 0)
            {
                printf("\e[31m[INFO]\e[m Not implemented yet.");
            }
            else
            {
                // apply temperature sampling TODO: review those sample strategy
                sample_top_k(&candidiate_p, generation_config.top_k, 1);
                sample_tail_free(&candidiate_p, generation_config.tfs_z, 1);
                sample_typical(&candidiate_p, generation_config.typical_p, 1);
                sample_top_p(&candidiate_p, generation_config.top_p, 1);
                sample_temperature(&candidiate_p, temperature);
                next_token_id = sample_token(&candidiate_p); // random sample as per probability
            }
        }
        PROFILE_END("[ Sampling ]");
        //  checks if the generated token is the end-of-sequence (EOS)
        // TODO: Qwen use distinct tokens from LLama
        if (next_token_id == config.eos_token_id)
        {
            printf("\e[31m[INFO]\e[m EOS detected\n");
            stop_generation_tolerance--; // a tolerance of the number of EOS. For instance, a successive occurence of EOSs means the end of the generation
            continue; // Do not update the context when meeting EOS            
        }
        // This checks if the generated token corresponds to the "padding" token (usually the token with ID 1).
        // Padding tokens are typically used to align sequences during batch processing but don't carry any meaningful content.
        if (next_token_id == config.pad_token_id)
        {
            printf("\e[31m[INFO]\e[m padding token detected\n");
            continue; // Do not update the context when meeting EOS            
        }
        stop_generation_tolerance = 2;



        // refresh the context TODO: another implementation, expanding the context vector progressively
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(next_token_id);
        generate_ids.push_back(next_token_id);
        // refresh the input_ids for next pass
        input_ids = std::vector<int>{next_token_id};
        if (interactive)
        {
            std::string output_text = tokenizer.decode(input_ids);
            std::cout << output_text << std::flush;
        }
        --n_remain;

        // printf("\e[31m[INFO]\e[m Context Length: (%zu / %d)\n", last_n_tokens.size(), max_context_length);
    }
    if (interactive) std::cout << std::endl;
    Profiler::getInstance().report_internal();
    Profiler::getInstance().reset();
    return generate_ids;
    
}