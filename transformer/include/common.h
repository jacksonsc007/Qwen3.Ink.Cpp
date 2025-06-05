#ifndef COMMON_H
#define COMMON_H
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cassert>

#include "model.h"

#define ASSERT(condition)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            std::cout << "Assertion failure: " << #condition << std::endl;                                             \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define MAX_LINEAR_LENGTH 1024 * 1024 * 16  // 16MB, TO BE REMOVED with better memory allocation!

#ifdef DEBUG
    #define IF_DEBUG(code) do { code } while (0)
#else
    #define IF_DEBUG(code) do { } while (0)
#endif

#ifdef DEBUG_IO
    #define IF_DEBUG_IO(code) do { code } while (0)
#else
    #define IF_DEBUG_IO(code) do { } while (0)
#endif

#ifdef DEBUG_ATTENTION
    #define IF_DEBUG_ATTENTION(code) do { code } while (0)
#else
    #define IF_DEBUG_ATTENTION(code) do { } while (0)
#endif

#ifdef DEBUG_DECODER_LAYER
    #define IF_DEBUG_DECODER_LAYER(code) do { code } while (0)
#else
    #define IF_DEBUG_DECODER_LAYER(code) do { } while (0)
#endif

#ifdef DEBUG_DECODER
    #define IF_DEBUG_DECODER(code) do { code } while (0)
#else
    #define IF_DEBUG_DECODER(code) do { } while (0)
#endif

#define QK 32

struct pack_q4_tensor {
    uint8_t qx[QK / 2];
    float scale;
};

struct pack_q8_tensor {
    int8_t qx[QK];
    float scale;
};

template <typename T>
class Matrix3D {
   public:
    Matrix3D(T *data, int dim_x, int dim_y, int dim_z) : m_data(data), m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z) {
        assert(m_dim_x >= 0 && m_dim_y >= 0 && m_dim_z >= 0);
    }
    
    Matrix3D(const Matrix3D<T> &other)
    {
        m_data = other.m_data;
        m_dim_x = other.m_dim_x;
        m_dim_y = other.m_dim_y;
        m_dim_z = other.m_dim_z;
    }

    Matrix3D& operator=(const Matrix3D<T> &other)
    {
        if (this == &other) return *this;
        m_data = other.m_data;
        m_dim_x = other.m_dim_x;
        m_dim_y = other.m_dim_y;
        m_dim_z = other.m_dim_z;
        return *this;
    }
    
    /*
    Reshape matrix without altering underlying memory layout
    */
    void view(const int new_x, int new_y, int new_z)
    {
       size_t new_size = (size_t) new_x * new_y * new_z;
       assert (size() == new_size);
       m_dim_x = new_x;
       m_dim_y = new_y;
       m_dim_z = new_z;
    }
    
    size_t size()
    {
        return (size_t)m_dim_x * m_dim_y * m_dim_z;
    }

    // Repeat along specified dimension
    Matrix3D repeat(int dim, int times, T* dest) const 
    {
        if (dim < 0 || dim > 2) {
            throw std::invalid_argument("Dimension must be 0 (x), 1 (y), or 2 (z)");
        }
        if (times < 1) {
            throw std::invalid_argument("Repeat times must be at least 1");
        }

        // Calculate new dimensions
        int new_dim_x = (dim == 0) ? m_dim_x * times : m_dim_x;
        int new_dim_y = (dim == 1) ? m_dim_y * times : m_dim_y;
        int new_dim_z = (dim == 2) ? m_dim_z * times : m_dim_z;

        Matrix3D result(dest, new_dim_x, new_dim_y, new_dim_z);
        // Fill the new matrix
        for (int x = 0; x < new_dim_x; ++x) 
        {
            for (int y = 0; y < new_dim_y; ++y) 
            {
                for (int z = 0; z < new_dim_z; ++z) 
                {
                    // Calculate original coordinates
                    int orig_x = (dim == 0) ? x / times  : x;
                    int orig_y = (dim == 1) ? y / times : y;
                    int orig_z = (dim == 2) ? z / times : z;
                    
                    result(x, y, z) = (*this)(orig_x, orig_y, orig_z);
                }
            }
        }
        return result;
    }

    T &operator()(int x, int y, int z) {
        if (x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
            printf("%d, %d, %d\n", x, y, z);
            printf("%d, %d, %d\n", m_dim_x, m_dim_y, m_dim_z);
            throw std::out_of_range("Matrix3D: Indices out of range.");
        }
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    const T &operator()(int x, int y, int z) const {
        if (x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
            printf("%d, %d, %d\n", x, y, z);
            printf("%d, %d, %d\n", m_dim_x, m_dim_y, m_dim_z);
            throw std::out_of_range("Matrix3D: Indices out of range.");
        }
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    bool operator==(const Matrix3D<T> &other) const {
        if (m_dim_x != other.m_dim_x || m_dim_y != other.m_dim_y || m_dim_z != other.m_dim_z) {
            return false;
        }

        for (int x = 0; x < m_dim_x; ++x) {
            for (int y = 0; y < m_dim_y; ++y) {
                for (int z = 0; z < m_dim_z; ++z) {
                    if ((*this)(x, y, z) != other(x, y, z)) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    int length() const { return m_dim_x * m_dim_y * m_dim_z; }
    T sum() const {
        T sum = 0;
        for (int i = 0; i < this->length(); i++) {
            sum += this->m_data[i];
        }
        return sum;
    }
    
    T mean() const{
        return this->sum() / this->length();
    }
    
    T max() const{
        T max = this->m_data[0];
        for (int i = 0; i < this->length(); i++) {
            if (this->m_data[i] > max) {
                max = this->m_data[i];
            }
        }
        return max;
    }
    
    T min() const{
        T min = this->m_data[0];
        for (int i = 0; i < this->length(); i++) {
            if (this->m_data[i] < min) {
                min = this->m_data[i];
            }
        }
        return min;
    }
    
    void statistics() const
    {
        std::cout << "max: " << this->max() << ", min: " << this->min() << ", mean: " << this->mean() << ", sum: " << this->sum() << std::endl; 
    }
    

    T sum(int size) const {
        T sum = 0;
        for (int i = 0; i < size; i++) {
            sum += this->m_data[i];
        }
        return sum;
    }

    T sum(int size, int start_idx) const {
        T sum = 0;
        for (int i = 0; i < size; i++) {
            sum += this->m_data[start_idx + i];
        }
        return sum;
    }

    void load(const char *path) {
        std::ifstream infile(path, std::ios::binary | std::ios::in);
        if (infile.fail()) {
            std::cout << strerror(errno) << ": " << path << std::endl;
            throw("failed to load");
        } else {
            infile.read(reinterpret_cast<char *>(this->m_data), this->length() * sizeof(T));
            infile.close();
        }
    }
    T *m_data;
    int m_dim_x, m_dim_y, m_dim_z;

    // Default constructor
    Matrix3D() { m_data = NULL; }
};

static inline void debug_info(std::string s) {
#ifdef DEBUG
    std::cout << s << std::endl;
#endif
}
#endif
