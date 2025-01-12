#include <iostream>
#include <chrono>
#include <vector>
#include <random>

void matvec(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) 
    {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) 
        {
            sum += B[i * N + j] * A[j];
        }
        C[i] = sum;
    }
}

void initialize_matrix(float* matrix, size_t size, float value) {
    if (value == -1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (size_t i = 0; i < size; ++i) {
            matrix[i] = dis(gen);
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            matrix[i] = value;
        }
    }
}

int main() {
    const size_t N = 35000;
    std::vector<float> A(N); // 1 * N
    std::vector<float> B(N * N); // N * N
    std::vector<float> C(N); // 1 * N

    // Initialize matrices
    initialize_matrix(A.data(), N, -1.0f);
    initialize_matrix(B.data(), N * N, -1.0f);
    initialize_matrix(C.data(), N, -1.0f);

    auto start = std::chrono::high_resolution_clock::now();
    int ITERS = 10;
    for (int i = 0; i < ITERS; ++i) {
        matvec(A.data(), B.data(), C.data(), N);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count(); // seconds

    // Calculate total data transferred
    double totalDataTransferred = (N * N + N) * sizeof(float); // bytes
    totalDataTransferred *= ITERS;
    double bandwidth = totalDataTransferred / duration; // bytes/second

    std::cout << "Execution Time: " << duration << " seconds\n";
    std::cout << "Measured Bandwidth: " << bandwidth / (1024 * 1024 * 1024) << " GB/s\n";

    return 0;
}