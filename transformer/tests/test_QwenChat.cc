#include <cstdint>
#include "Generate.h"
#include "QwenForCausalLM.h"
#include "QwenOperator.h"
#include "common.h"
#include "model.h"

// std::map<std::string, int> model_config = 
// {
//     {"qwen3-8b", 0}
// };

// std::map<std::string, std::string> model_path_config =
// {
//     {"qwen3-8b", "model_weights/fp32/qwen3_8b"},
// };

enum LinearKernelType{
    Afp32Wfp32 = 0,
    A80W40 = 1,
    KernelCount = 2
};


struct model_meta{
    std::string model_name;
    std::string model_path;
    int data_type;
};

struct model_meta MODEL_REPOSITORY[KernelCount] = {
    [Afp32Wfp32] = {
        .model_name = "qwen3-8b",
        .model_path = "model_weights/fp32/qwen3_8b",
    },
    [A80W40] = {
        .model_name = "qwen3-8b-A80W40",
        .model_path = "model_weights/int4/qwen3_8b_A80W40",
    },    
};


int main(int argc, char** argv)
{
    std::string target_model = "qwen3-8b-A80W40";
    LinearKernelType kernel_type = A80W40;

    
    printf("\e[31m[INFO]\e[m Loading Model ...\n");
    std::string model_path = MODEL_REPOSITORY[kernel_type].model_path;
    int data_format = MODEL_REPOSITORY[kernel_type].data_type;
    qwen_params generation_config {
        -1,
        1,
        128
    };
    if (kernel_type == A80W40)
    {
        int bs = 1;
        int max_sqlen = 4096;
        int qk;  // group size
        assert (bs == 1); // only support bs = 1
        qwen3_config config(bs,max_sqlen);
        Qwen3ForCausalLM model = Qwen3ForCausalLM(model_path, config);
    printf("\e[32m[INFO]\e[m Model Loaded ...\n");
        std::string tiktoken_path = "qwen-7b-chat/qwen.tiktoken";
        
        #ifdef DEBUG
            std::string input = "Give me a short introduction to large language model.";
            QwenGenerate(&model, input, generation_config, tiktoken_path, true, config);
        #else
        while(true){
            std::cout << "USER: ";
            std::string input;
            std::getline(std::cin, input);
            input = "A chat between a human and an assistant.\n\n### Human: " + input + "\n### Assistant: \n";
            QwenGenerate(&model, input, generation_config, tiktoken_path, true, config);

        }
        #endif

        
    }
    else
    {
        std::cerr << "Not Implemented" << std::endl;
        throw("Datatype Not Implemented");

    }

    
}