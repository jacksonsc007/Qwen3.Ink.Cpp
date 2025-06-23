#pragma  once
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <string>

// quantization block size
#define QK 32
// Define alignment boundary (e.g., 64-byte for cache-line optimization)
#define ALIGNMENT 64


enum LinearKernelType { Afp32Wfp32 = 0, A80W40 = 1, A81W41 = 2, KernelCount = 3 };

struct model_meta {
    std::string model_name;
    std::string model_path;
    int data_type;
};


struct qwen3_config {
    int batchsize;
    int num_key_value_heads;
    int num_layers;
    int max_sqlen;
    int num_q_head, num_kv_head;

    int hidden_dim;
    int head_dim;
     
    int vocsize;
    int qk;  // group size
    // for tokenizer
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    int im_start_id;
    int im_end_id;

    qwen3_config() = delete;

    qwen3_config(int batch, int max_sqlen)
        : batchsize(batch),
          max_sqlen(max_sqlen)
    {
        vocsize = 151936;
        num_q_head = 32;
        num_kv_head = 8;
        num_key_value_heads = 8; // GQA

        hidden_dim = 4096;
        head_dim = 128;

        num_layers = 36;
        bos_token_id = 151643;
        eos_token_id = 151645;
        pad_token_id = 151643;
        im_start_id = 151644;
        im_end_id = 151645;
    }
};

// Generation parameters structure
struct qwen_params {
    int32_t n_predict = 4096;  // new tokens to predict
    int32_t n_ctx = 512;      // context size
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
