#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>

#define N 1024 // Matrix size
// Blocked matrix multiplication to optimize memory access
//
void matrixMultiplyBlocked(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C, long long& totalFLOPs, long long& totalBytes, int blockSize) {
    totalFLOPs = 0;
    totalBytes = 0;

    for (int i = 0; i < N; i += blockSize) {
        for (int j = 0; j < N; j += blockSize) {
            for (int k = 0; k < N; k += blockSize) {
                // Perform block multiplication
                for (int ii = i; ii < std::min(i + blockSize, N); ++ii) {
                    for (int jj = j; jj < std::min(j + blockSize, N); ++jj) {
                        for (int kk = k; kk < std::min(k + blockSize, N); ++kk) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                            totalFLOPs += 2; // 1 multiplication + 1 addition
                        }
                    }
                }
            }
        }
    }

    // Memory bandwidth for blocked version (similar to naive, but with smaller blocks)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            totalBytes += sizeof(float) * 3; // 2 reads (A and B) and 1 write (C)
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

    // Measure the time taken to multiply matrices (Blocked)
    long long totalFLOPs = 0, totalBytes = 0;
    int blockSize = 64; // Block size for the blocked matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyBlocked(A, B, C, totalFLOPs, totalBytes, blockSize);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Blocked Matrix Multiplication (Block size " << blockSize << ") took " << duration.count() << " seconds.\n";
    printMetrics(totalFLOPs, totalBytes, duration.count());

    return 0;
}