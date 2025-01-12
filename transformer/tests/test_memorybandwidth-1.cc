#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>

#define N 1024 // Matrix size

// Function to multiply two matrices A and B, and store the result in C
void matrixMultiplyNaive(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C, long long& totalFLOPs, long long& totalBytes) {
    totalFLOPs = 0;
    totalBytes = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
                totalFLOPs += 2; // 1 multiplication + 1 addition
            }
            totalBytes += sizeof(float) * 2; // 2 reads (A and B)
            totalBytes += sizeof(float); // 1 write (C)
        }
    }
}

// Function to print the performance metrics
void printMetrics(long long totalFLOPs, long long totalBytes, double duration) {
    double bandwidth = totalBytes / (duration * 1e9); // GB/s
    double flops = totalFLOPs / (duration * 1e9); // GFLOP/s
    std::cout << "FLOPs: " << flops << " GFLOP/s\n";
    std::cout << "Memory Bandwidth: " << bandwidth << " GB/s\n";
}

int main() {
    // Initialize matrices A, B, and C with random values
    std::vector<std::vector<float>> A(N, std::vector<float>(N));
    std::vector<std::vector<float>> B(N, std::vector<float>(N));
    std::vector<std::vector<float>> C(N, std::vector<float>(N));

    // Random initialization
    srand(42); // For reproducibility
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = static_cast<float>(rand()) / RAND_MAX;
            B[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Measure the time taken to multiply matrices (Naive)
    long long totalFLOPs = 0, totalBytes = 0;
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyNaive(A, B, C, totalFLOPs, totalBytes);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Naive Matrix Multiplication took " << duration.count() << " seconds.\n";
    printMetrics(totalFLOPs, totalBytes, duration.count());

    return 0;
}