#include "Generate.h"
#include "Int4QwenForCausalLM.h"

std::map<std::string, int> model_config = 
{
    {"qwen-7b-chat", 0}
};

std::map<std::string, int> data_format_config = 
{
    {"FP32", FP32},
    {"INT8", INT8},
    {"INT4", INT4},
};

std::map<std::string, std::string> model_path_config =
{
    {"qwen-7b-chat", "../INT4/models/qwen-7b-chat"},
};


int main(int argc, char** argv)
{
    std::string target_model = "qwen-7b-chat";
    std::string target_data_type = "FP32";

    if (argc == 3)
    {
        target_model = argv[1];
        target_data_type = argv[2];
        if (model_config.count(target_model) == 0)
        {
            std::cerr << "Model " << target_model << " not supported." << std::endl;
            std::cerr << "Please choose from the following models: " << std::endl;
            for (auto &m : model_config)
            {
                std::cerr << m.first << std::endl;
            }
            throw("Unsupport model\n");
        }
        if (data_format_config.count(target_data_type)==0)
        {
            std::cerr << "Data type " << target_data_type << " not supported." << std::endl;
            std::cerr << "Please choose from the following Data type: " << std::endl;
            for (auto &m : data_format_config)
            {
                std::cerr << m.first << std::endl;
            }
            throw("Unsupport data type\n");
            
        }
    }
    else if (argc == 1)
    {
        printf("\e[31m[INFO]\e[m Selecting Default Model: %s\n", target_model.c_str());
        printf("\e[31m[INFO]\e[m Selecting Default Data Type: %s\n", target_data_type.c_str());
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " [model] [data_type]" << std::endl;
        std::cerr << "Please choose from the following models: " << std::endl;
        for (auto &m : model_config)
        {
            std::cerr << m.first << std::endl;
        }
        std::cerr << "Please choose from the following Data type:" << std::endl;
        for (auto &m : data_format_config)
        {
            std::cerr << m.first << std::endl;
        }
    }
    
    printf("\e[31m[INFO]\e[m Loading Model ...\n");
    std::string model_path = model_path_config[target_model];
    qwen_params generation_config;
    if (data_format_config[target_data_type] == FP32)
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
        qwen_config config(bs, num_heads, num_layers, max_sqlen, embed_dim, hidden_dim, vocsize, padding_idx);
        Int4QwenForCausalLM model = Int4QwenForCausalLM(model_path, config);
    printf("\e[32m[INFO]\e[m Model Loaded ...\n");
        std::string tiktoken_path = "../qwen-7b-chat/qwen.tiktoken";
        
        while(true){
            std::cout << "USER: ";
            std::string input;
            std::getline(std::cin, input);
            input = "A chat between a human and an assistant.\n\n### Human: " + input + "\n### Assistant: \n";
            QwenGenerateInt4(&model, input, generation_config, tiktoken_path, true, config);

        }

        
    }
    else
    {
        std::cerr << "Not Implemented" << std::endl;
        throw("Datatype Not Implemented");

    }

    
}