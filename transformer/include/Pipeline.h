#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "operators.h"
#include "utils.h"
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

// Generation parameters structure
struct qwen_params {
    int32_t seed = -1;        // RNG seed
    int32_t n_threads = 1;    // TODO: fix this
    int32_t n_predict = 4096;  // new tokens to predict
    int32_t n_parts = -1;     // amount of model parts (-1 = determine from model dimensions)
    int32_t n_ctx = 512;      // context size
    int32_t n_batch = 512;    // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep = 0;       // number of tokens to keep from initial prompt
    int32_t n_vocab = 151936;  // vocabulary size

    // sampling parameters
    std::unordered_map<int, float> logit_bias;  // logit bias for specific tokens
    int32_t top_k = 40;                         // <= 0 to use vocab size
    float top_p = 0.95f;                        // 1.0 = disabled
    float tfs_z = 1.00f;                        // 1.0 = disabled
    float typical_p = 1.00f;                    // 1.0 = disabled
    float temp = 0.80f;                         // 1.0 = disabled
    float repeat_penalty = 1.10f;               // 1.0 = disabled
    int32_t repeat_last_n = 64;                 // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float frequency_penalty = 0.00f;            // 0.0 = disabled
    float presence_penalty = 0.00f;             // 0.0 = disabled
    int mirostat = 0;                           // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float mirostat_tau = 5.00f;                 // target entropy
    float mirostat_eta = 0.10f;                 // learning rate
};

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

    std::vector<int> generate(const std::string& text, const qwen_params& generation_config);
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
}; 