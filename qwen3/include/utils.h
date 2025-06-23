#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <cstdlib>
#include "profiler.h"

#define ASSERT(condition)                                                  \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::cout << "Assertion failure: " << #condition << std::endl; \
            abort();                                                       \
        }                                                                  \
    } while (0)

#define MAX_LINEAR_LENGTH 1024 * 1024 * 16  // 16MB, TO BE REMOVED with better memory allocation!

#ifdef DEBUG
#define IF_DEBUG(code) \
    do {               \
        code;          \
    } while (0);
#else
#define IF_DEBUG(code) \
    do {               \
    } while (0);
#endif

#ifdef DEBUG_IO
#define IF_DEBUG_IO(code) \
    do {                  \
        code              \
    } while (0)
#else
#define IF_DEBUG_IO(code) \
    do {                  \
    } while (0)
#endif

#ifdef DEBUG_ATTENTION
#define IF_DEBUG_ATTENTION(code) \
    do {                         \
        code                     \
    } while (0);
#else
#define IF_DEBUG_ATTENTION(code) \
    do {                         \
    } while (0);
#endif

#ifdef DEBUG_DECODER_LAYER
#define IF_DEBUG_DECODER_LAYER(code) \
    do {                             \
        code                         \
    } while (0)
#else
#define IF_DEBUG_DECODER_LAYER(code) \
    do {                             \
    } while (0)
#endif

#ifdef DEBUG_DECODER
#define IF_DEBUG_DECODER(code) \
    do {                       \
        code                   \
    } while (0)
#else
#define IF_DEBUG_DECODER(code) \
    do {                       \
    } while (0)
#endif
#define STATS_START(x) Profiler::getInstance().start(x)
#define STATS_FLOPS(x, y) Profiler::getInstance().start(x, y)
#define STATS_END(x) Profiler::getInstance().stop(x)

#ifdef PROFILER
#define PROFILE_START(x) Profiler::getInstance().start(x)
#define PROFILE_START_FLOPS(x, y) Profiler::getInstance().start(x, y)
#define PROFILE_END(x) Profiler::getInstance().stop(x)
#else
#define PROFILE_START(x)
#define PROFILE_START_FLOPS(x, y)
#define PROFILE_END(x)
#endif

#define MAX_SQ_ERROR_MAX 5e-5
#define ERROR_MAX 1e-9
#define INT_ERROR_MAX 1e-5

template <typename T>
void read_to_array(const char* path, T* array, size_t size);

template <typename T>
void write_array_to_file(const char* path, T* array, size_t size);

template <typename T>
bool check_two_equal(T* array, T* array2, int size);

template <>
bool check_two_equal(int8_t* array, int8_t* array2, int size);

bool check_two_equal(int8_t* array, int8_t* array2, int size, float error);

bool check_two_equal(float* array, float* array2, int size, float error);
bool check_two_exact_equal(int8_t* array, int8_t* array2, int size);
void print_MSE_max_diff(float* a, float* a2, int size);

void print_first_k_elelment(std::string name, const int8_t* arr, int k, int start_idx = 0);
void print_first_k_elelment(std::string name, const int32_t* arr, int k, int start_idx = 0);
void print_first_k_elelment(std::string name, const float* arr, int k, int start_idx = 0);

#ifdef QM_METAL
template <typename T>
void allocate_aligned_memory(T*& ptr, size_t size);
#else
template <typename T>
void allocate_aligned_memory(T*& ptr, size_t size);
#endif

void deallocate_memory(void* ptr);

#endif
