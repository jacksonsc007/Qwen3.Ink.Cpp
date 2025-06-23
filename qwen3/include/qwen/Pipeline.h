#pragma once

#include <cassert>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include "operators.h"
#include "utils.h"
#include "QwenConfig.h"
#include "QwenTokenizer.h"
#include "QwenForCausalLM.h"

// Random number generator for sampling
inline std::mt19937 random_generator;

// Token data structure for sampling
typedef struct token_data {
    int id;       // token id
    float logit;  // log-odds of the token
    float p;      // probability of the token
} token_data;

typedef struct token_data_array {
    token_data* data;
    size_t size;
    bool sorted;
} token_data_array;


// Sampling function declarations
void sample_repetition_penalty(token_data_array* candidates, const int* last_tokens, size_t last_tokens_size,
                               float penalty);

void sample_frequency_and_presence_penalties(token_data_array* candidates, const int* last_tokens_p,
                                             size_t last_tokens_size, float alpha_frequency, float alpha_presence);

int sample_token_greedy(token_data_array* candidates);

void sample_temperature(token_data_array* candidates_p, float temp);

void sample_softmax(token_data_array* candidates);

int sample_token(token_data_array* candidates);

void sample_top_k(token_data_array* candidates, int k, size_t min_keep);

int sample_token_mirostat(const int n_vocab, token_data_array* candidates, float tau, float eta, int m, float* mu);

int sample_token_mirostat_v2(token_data_array* candidates, float tau, float eta, float* mu);

void sample_tail_free(token_data_array* candidates, float z, size_t min_keep);

void sample_typical(token_data_array* candidates, float p, size_t min_keep);

void sample_top_p(token_data_array* candidates, float p, size_t min_keep);

// Pipeline class for text generation
class Pipeline {
private:
    QwenTokenizer tokenizer;
    Qwen3ForCausalLM* model;
    qwen3_config config;
    std::string tiktoken_path;
    bool interactive;

public:
    Pipeline(void* model_ptr, const std::string& tiktoken_path, bool interactive, const qwen3_config& config);
    ~Pipeline();

    std::vector<int> generate(const std::vector<std::string>& history, const qwen_params& generation_config);
    std::string decode(const std::vector<int>& tokens);
    std::vector<int> encode(const std::string& text, int max_length);

    // Static sampling methods
    static void sample_repetition_penalty(token_data_array* candidates, const int* last_tokens, size_t last_tokens_size,
                               float penalty);
    static void sample_frequency_and_presence_penalties(token_data_array* candidates, const int* last_tokens_p,
                                             size_t last_tokens_size, float alpha_frequency, float alpha_presence);
    static int sample_token_greedy(token_data_array* candidates);
    static void sample_temperature(token_data_array* candidates_p, float temp);
    static void sample_softmax(token_data_array* candidates);
    static int sample_token(token_data_array* candidates);
    static void sample_top_k(token_data_array* candidates, int k, size_t min_keep);
    static int sample_token_mirostat(const int n_vocab, token_data_array* candidates, float tau, float eta, int m, float* mu);
    static int sample_token_mirostat_v2(token_data_array* candidates, float tau, float eta, float* mu);
    static void sample_tail_free(token_data_array* candidates, float z, size_t min_keep);
    static void sample_typical(token_data_array* candidates, float p, size_t min_keep);
    static void sample_top_p(token_data_array* candidates, float p, size_t min_keep);

private:
    // Helper for the main generation loop, takes already-encoded input_ids
    std::vector<int> _generate_from_ids(std::vector<int> input_ids, const qwen_params& generation_config);

    // Encapsulate the sampling process
    int sample_next_token(std::vector<float>& logits, std::vector<int>& last_n_tokens, const qwen_params& generation_config);
}; 