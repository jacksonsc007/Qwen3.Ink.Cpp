#include <cmath>

#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"
#include "Int4QwenOperator.h"
// #include <sstream>

void test_FPLinear_int4() {
    const int m = 1, n = 32000, k = 4096;

    MemoryAllocator mem_buf;

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(m * k), 1, m, k);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(n * k), 1, n, k);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> output(mem_buf.get_fpbuffer(m * n), 1, m, n);

    hidden_states.load("../tests/assets/input.bin");
    outputGT.load("../tests/assets/output.bin");

    // quantize the weight to int4
    Matrix3D<uint8_t> int4_weight((uint8_t *)mem_buf.get_int8buffer(n * k / 2), 1, n, k / 2);
    // Linear_FP_int4 int4_op;
    Linear_FP_int4 int4_op = Linear_FP_int4(int4_weight, "../INT4/models/LLaMA_7B_2_chat/lm_head/");

    Matrix3D<float> outputQ(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> outputQ_simd(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> outputQ_fast(mem_buf.get_fpbuffer(m * n), 1, m, n);

    // warm up
    for (int i = 0; i < 1; i++) {
        int4_op.forward(hidden_states, outputQ_fast);
    }

    const int flops = k * m * n * 2;
    int4_op.forward_ref(hidden_states, outputQ);

    for (int i = 0; i < 50; i++) {
        STATS_FLOPS(int4_op.profile_name, flops);
        int4_op.forward(hidden_states, outputQ_fast);
        STATS_END(int4_op.profile_name);
    }
    bool success = check_two_equal(outputQ.m_data, outputQ_fast.m_data, outputQ_fast.length(), 1e-3);

    if (!success) {
        std::cout << "-------- Sanity check of " << int4_op.profile_name << " implementation: Fail! -------- "
                  << std::endl;
        exit(-1);
    } else
        std::cout << "-------- Sanity check of " << int4_op.profile_name << " implementation: Passed! -------- "
                  << std::endl;
}

void initialize_matrix(float * matrix, size_t size, float value){
    if ( value == -1.0 )
    {
        // printf("randomly initilize \n");
        for (size_t i = 0; i < size; i++){
            matrix[i] = (float) rand() / (float) RAND_MAX;
        } 
    }
    else
    {
        // printf("initilize to %f \n", value);
        for (size_t i = 0; i < size; i++){
            matrix[i] = value;
        } 
    }
}

void benchmark(long long rows, long long hidden, long long cols, int test_iters)
{
    long long m = rows, n = cols, k = hidden;
    printf("\e[31m[INFO]\e[m m=%lld, n=%lld, k=%lld\n", m, n, k);

    MemoryAllocator mem_buf;

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(m * k), 1, m, k);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(n * k), 1, n, k);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> output(mem_buf.get_fpbuffer(m * n), 1, m, n);

    initialize_matrix(hidden_states.m_data, hidden_states.length(), -1);
    initialize_matrix(weight.m_data, weight.length(), -1);
    
    
    
    Linear_FP fp32_op = Linear_FP(weight);
    int warming_up_iters = 5;
    int benchmark_iters = test_iters;
    // warming up and test if the output match the gt
    for (int i = 0; i < warming_up_iters; i++) {
        // printf("\e[31m[INFO]\e[m Warming up\n");
        fp32_op.forward(hidden_states, output);
    }
    // start benchmark
    long long flops = k * m * n * 2;
    std::ostringstream os;
    os << "[" << m << "x" << n << "x" << k << "]";
    std::string profile_name = os.str();
    for (int i = 0; i < benchmark_iters; i++) {
        STATS_FLOPS(profile_name, flops);
        fp32_op.forward(hidden_states, output);
        STATS_END(profile_name);
    }
    
    // memory deallocate
    free(hidden_states.m_data);
    free(weight.m_data);
    free(outputGT.m_data);
    free(output.m_data);

}

void integrity_check(long long rows, long long hidden, long long cols, std::string module_name, int default_benchmark_iters)
{
    long long m = rows, n = cols, k = hidden;
    printf("\e[31m[INFO]\e[m m=%lld, n=%lld, k=%lld\n", m, n, k);
    MemoryAllocator mem_buf;

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(m * k), 1, m, k);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(n * k), 1, n, k);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> output(mem_buf.get_fpbuffer(m * n), 1, m, n);

    std::string workspace = "../assets/";
    std::string input_path = workspace + "input_" + module_name + ".bin";
    std::string gt_output_path = workspace + "output_" + module_name + ".bin";
    std::string weight_path = workspace + "weight_" + module_name + ".bin";
    hidden_states.load(input_path.c_str());
    outputGT.load(gt_output_path.c_str());
    
    Linear_FP fp32_op = Linear_FP(weight, weight_path.c_str());
    int warming_up_iters = 1;
    int benchmark_iters = 0;
    if (default_benchmark_iters == -1)
        benchmark_iters = 100;
    else benchmark_iters = default_benchmark_iters;
    // warming up and test if the output match the gt
    for (int i = 0; i < warming_up_iters; i++) {
        // printf("\e[31m[INFO]\e[m Warming up");
        fp32_op.forward(hidden_states, output);
        bool success = check_two_equal(outputGT.m_data, output.m_data, output.length());
        assert (success);
    }
    // start benchmark
    long long flops = k * m * n * 2;
    std::ostringstream os;
    os << "[" << m << "x" << n << "x" << k << "]-";
    os << module_name.c_str();
    std::string profile_name = os.str();
    for (int i = 0; i < benchmark_iters; i++) {
        STATS_FLOPS(profile_name, flops);
        fp32_op.forward(hidden_states, output);
        STATS_END(profile_name);
    }

    bool success = check_two_equal(outputGT.m_data, output.m_data, output.length());

    if (!success) {
        printf("\033[1;31m");
        printf("-------- Sanity check of %s implementation: Fail! -------- \n",fp32_op.profile_name.c_str());
    } 
    else
    {
        printf("\033[1;32m");
        printf("-------- Sanity check of %s implementation: Passed! -------- \n",fp32_op.profile_name.c_str());
        printf("\e[32m[Congrats]\e[m -------- Sanity check of %s implementation: Passed! -------- \n", fp32_op.profile_name.c_str());
    }
    // memory deallocate
    free(hidden_states.m_data);
    free(weight.m_data);
    free(outputGT.m_data);
    free(output.m_data);
}
void test_FPLinear_fp32() {
    printf("\e[31m[INFO]\e[m Start testing FP32 matmul performance \n");
    benchmark(1, 4096, 4096, 20);
    // benchmark(1, 4096, 11008, 20);
    // benchmark(1, 4096, 151936, 20);
    // // benchmark(4096, 4096, 4096, 1);
    // printf("\e[31m[INFO]\e[m Test the prefilling stage \n");
    // benchmark(19, 4096, 4096, 20);
    // integrity_check(19, 4096, 4096,"o_proj", -1);
    // integrity_check(19, 4096, 151936, "lm_head", 5);
}

void test_qwen_linear_int4() {
    // use lm_head as an temporary example
    std::string workspace = "../../INT4/models/qwen-7b-chat/lm_head/";
    const int weight_dim_x = 5;
    const int weight_dim_y = 151936;
    const int weight_dim_z = 4096;
    printf("\e[31m[INFO]\e[m [%d x %d x %d]\n", weight_dim_x, weight_dim_y, weight_dim_z);

    const int m = weight_dim_x, n = weight_dim_y, k = weight_dim_z;

    MemoryAllocator mem_buf;

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(m * k), 1, m, k);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> output(mem_buf.get_fpbuffer(m * n), 1, m, n);
    // random initialize input
    initialize_matrix(hidden_states.m_data, hidden_states.length(), -1.0);

    Qwen_Linear_with_bias_Int4 matmul_op = Qwen_Linear_with_bias_Int4(workspace, weight_dim_x, weight_dim_y, weight_dim_z);

    print_first_k_elelment("outputGT", outputGT.m_data, 10);
    print_first_k_elelment("output", output.m_data, 10);

    matmul_op.forward(hidden_states, output);
    matmul_op.forward_reference(hidden_states, outputGT);

    print_first_k_elelment("outputGT", outputGT.m_data, 10);
    print_first_k_elelment("output", output.m_data, 10);
    
    bool success = check_two_equal(outputGT.m_data, output.m_data, outputGT.length());

    if (!success) {
        printf("\033[1;31m");
        printf("-------- Sanity check of %s implementation: Fail! -------- \n",matmul_op.profile_name.c_str());
        exit(-1);
    } 
    printf("\033[1;32m");
    printf("-------- Sanity check of %s implementation: Passed! -------- \n",matmul_op.profile_name.c_str());
    
    // start benchmark
    long long flops = (long long) k * (long long)m * (long long)n * 2;
    int benchmark_iters = 40;
    printf("\e[31m[INFO]\e[m total flops: %lld\n", flops * benchmark_iters);
    std::string module_name = "Int4";
    std::ostringstream os;
    os << "[" << m << "x" << n << "x" << k << "]-";
    os << module_name.c_str();
    std::string profile_name = os.str();
    for (int i = 0; i < benchmark_iters; i++) {
        STATS_FLOPS(profile_name, flops);
        matmul_op.forward(hidden_states, output);
        STATS_END(profile_name);
    }

}
int main() {
    // test_FPLinear_int4();
    // test_FPLinear_fp32();
    test_qwen_linear_int4();
    Profiler::getInstance().report_internal();
}
