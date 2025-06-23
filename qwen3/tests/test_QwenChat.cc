#include <cstdint>

#include "Pipeline.h"
#include "QwenForCausalLM.h"
#include "QwenOperator.h"
#include "common.h"
#include "QwenConfig.h"

// std::map<std::string, int> model_config =
// {
//     {"qwen3-8b", 0}
// };

// std::map<std::string, std::string> model_path_config =
// {
//     {"qwen3-8b", "model_weights/fp32/qwen3_8b"},
// };

enum LinearKernelType { Afp32Wfp32 = 0, A80W40 = 1, A81W41 = 2, KernelCount = 3 };

struct model_meta {
    std::string model_name;
    std::string model_path;
    int data_type;
};

struct model_meta MODEL_REPOSITORY[KernelCount] = {
    [Afp32Wfp32] =
        {
            .model_name = "qwen3-8b",
            .model_path = "model_weights/fp32/qwen3_8b",
        },
    [A80W40] =
        {
            .model_name = "qwen3-8b-A80W40",
            .model_path = "model_weights/int4/qwen3_8b_A80W40",
        },
    [A81W41] =
        {
            .model_name = "qwen3-8b-A81W41",
            .model_path = "model_weights/int4/qwen3-8b-A81W41",
        },
};

int main(int argc, char** argv) {
    std::string target_model = "qwen3-8b-A81W41";
    LinearKernelType kernel_type = A81W41;

    // NOTE: we must call ggml_init before invoking GGML_FP16_TO_FP32
    const int ctx_size = 0;
    struct ggml_init_params params = {/*.mem_size   =*/ctx_size,
                                      /*.mem_buffer =*/NULL,
                                      /* no_alloc   =*/0};

    struct ggml_context* ctx;
    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return 1;
    }

    printf("\e[31m[INFO]\e[m Loading Model ...\n");
    std::string model_path = MODEL_REPOSITORY[kernel_type].model_path;
    int data_format = MODEL_REPOSITORY[kernel_type].data_type;
    qwen_params generation_config{-1, 1, 128};

    std::cout << R"(   ____                         _____      _____         _          ___    ___    ___ )" << '\n'
              << R"(  /___ \__      __  ___  _ __  |___ /      \_   \ _ __  | | __     / __\  / _ \  / _ \)" << '\n'
              << R"( //  / /\ \ /\ / / / _ \| '_ \   |_ \       / /\/| '_ \ | |/ /    / /    / /_)/ / /_)/)" << '\n'
              << R"(/ \_/ /  \ V  V / |  __/| | | | ___) | _ /\/ /_  | | | ||   <  _ / /___ / ___/ / ___/ )" << '\n'
              << R"(\___,_\   \_/\_/   \___||_| |_||____/ (_)\____/  |_| |_||_|\_\(_)\____/ \/     \/     )" << '\n'
              << '\n';
    std::cout
        << "Welcome! This project is study-oriented, please check more info on the homepage. \n"
        << "\n";
    if (kernel_type == A81W41) {
        int bs = 1;
        int max_sqlen = 4096;
        int qk;           // group size
        assert(bs == 1);  // only support bs = 1
        qwen3_config config(bs, max_sqlen);
        Qwen3ForCausalLM model = Qwen3ForCausalLM(model_path, config);
        printf("\e[32m[INFO]\e[m Model Loaded ...\n");
        std::string tiktoken_path = "tokenizers/qwen.tiktoken";

        Pipeline pipeline(&model, tiktoken_path, true, config);

#ifdef DEBUG
        std::string input = "Give me a short introduction to large language model.";
        pipeline.generate({input}, generation_config);
#else
        while (true) {
            std::cout << "USER: ";
            std::string input;
            std::getline(std::cin, input);
            pipeline.generate({input}, generation_config);
        }
#endif

    } else {
        std::cerr << "Not Implemented" << std::endl;
        throw("Datatype Not Implemented");
    }
}