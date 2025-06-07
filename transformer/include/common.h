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
#include "utils.h"

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
    #define IF_DEBUG(code) do { code } while (0);
#else
    #define IF_DEBUG(code) do { } while (0);
#endif

#ifdef DEBUG_IO
    #define IF_DEBUG_IO(code) do { code } while (0)
#else
    #define IF_DEBUG_IO(code) do { } while (0)
#endif

#ifdef DEBUG_ATTENTION
    #define IF_DEBUG_ATTENTION(code) do { code } while (0);
#else
    #define IF_DEBUG_ATTENTION(code) do { } while (0);
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
private:

public:
    std::unique_ptr<T[]> m_data;
    int m_dim_x, m_dim_y, m_dim_z;

    // Default constructor - creates empty matrix
    Matrix3D() : m_data(nullptr), m_dim_x(0), m_dim_y(0), m_dim_z(0) {}

    // Constructor with dimensions - allocates memory
    Matrix3D(int dim_x, int dim_y, int dim_z) 
        : m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z) 
    {
        assert(m_dim_x >= 0 && m_dim_y >= 0 && m_dim_z >= 0);
        if (size() > 0) {
            m_data = std::make_unique<T[]>(size());
        }
    }

    // Option 1: Safe version that always copies data
    Matrix3D(const T* data, int dim_x, int dim_y, int dim_z)
        : Matrix3D(dim_x, dim_y, dim_z) 
    {
        if (data && size() > 0) {
            std::copy(data, data + size(), m_data.get());
        }
    }

    // Option 2: Separate constructor for external data (NOT RECOMMENDED)
    // Only use if you absolutely need to wrap external memory
    Matrix3D(T* external_data, int dim_x, int dim_y, int dim_z, std::false_type /*dummy*/)
        : m_data(external_data, [](T*){}), // Custom deleter that does nothing
          m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z)
    {
        if (!external_data) {
            throw std::invalid_argument("External data pointer cannot be null");
        }
    }

    // Copy constructor
    Matrix3D(const Matrix3D<T>& other) 
        : m_dim_x(other.m_dim_x), m_dim_y(other.m_dim_y), m_dim_z(other.m_dim_z)
    {
        m_data = std::make_unique<T[]>(other.size());
        std::copy(other.data(), other.data() + other.size(), m_data.get());
    }

    // Move constructor
    Matrix3D(Matrix3D<T>&& other) noexcept
        : m_data(std::move(other.m_data)),
          m_dim_x(other.m_dim_x),
          m_dim_y(other.m_dim_y),
          m_dim_z(other.m_dim_z) 
    {
        other.m_dim_x = 0;
        other.m_dim_y = 0;
        other.m_dim_z = 0;
    }

    // Copy assignment
    Matrix3D& operator=(const Matrix3D<T>& other) {
        if (this == &other) return *this;
        
        // Create temporary and swap
        Matrix3D temp(other);
        swap(*this, temp);
        return *this;
    }

    // Move assignment
    Matrix3D& operator=(Matrix3D<T>&& other) noexcept {
        if (this == &other) return *this;
        
        m_data = std::move(other.m_data);
        m_dim_x = other.m_dim_x;
        m_dim_y = other.m_dim_y;
        m_dim_z = other.m_dim_z;
        
        other.m_dim_x = 0;
        other.m_dim_y = 0;
        other.m_dim_z = 0;
        
        return *this;
    }

    // Swap function for copy-and-swap idiom
    friend void swap(Matrix3D<T>& first, Matrix3D<T>& second) noexcept {
        using std::swap;
        swap(first.m_data, second.m_data);
        swap(first.m_dim_x, second.m_dim_x);
        swap(first.m_dim_y, second.m_dim_y);
        swap(first.m_dim_z, second.m_dim_z);
    }
    
    Matrix3D bind(T * data, int x, int y, int z)
    {
        Matrix3D output;
        return output;
    }

    // concatenate two matrix along x dim
    Matrix3D cat(Matrix3D other)
    {
        assert(other.m_dim_y = m_dim_y);
        assert(other.m_dim_z = m_dim_z);
        Matrix3D output = Matrix3D(other.m_dim_x + m_dim_x, m_dim_y, m_dim_z);
        std::copy(m_data.get(), m_data.get() + size(), output.data());
        std::copy(other.data(), other.data() + other.size(), output.data() + size());
        return output;
    }
    
    // exchange dimension 0 and 1
    Matrix3D permute01() {
        PROFILE_START("QwenAttention::permute");
        const int dim_x = m_dim_y;  // Swapped dimensions
        const int dim_y = m_dim_x;
        const int dim_z = m_dim_z;
        Matrix3D after(dim_x, dim_y, dim_z);
        
        const int outer_stride = m_dim_y * m_dim_z;
        const int inner_stride = m_dim_z;
        
        T* dest_ptr = after.m_data.get();
        const T* src_ptr = this->m_data.get();
        
        for (int i = 0; i < dim_x; i++) {
            for (int j = 0; j < dim_y; j++) {
                // Contiguous memory copy for the innermost dimension
                const int src_offset = j * outer_stride + i * inner_stride;
                const int dest_offset = i * dim_y * dim_z + j * dim_z;
                
                std::copy(src_ptr + src_offset,
                         src_ptr + src_offset + dim_z,
                         dest_ptr + dest_offset);
            }
        }
        
        PROFILE_END("QwenAttention::permute");
        return after;
    }
    
    /*
    Creates a new Matrix3D with the same shape (dimensions) as another matrix
    The data is uninitialized unless T has a default constructor
    */
    Matrix3D as_shape(const Matrix3D<T>& other) const {
        return Matrix3D(other.m_dim_x, other.m_dim_y, other.m_dim_z);
    }

    /*
    Creates a new Matrix3D with the same shape (dimensions) as this matrix
    The data is uninitialized unless T has a default constructor
    */
    Matrix3D as_shape() const {
        return Matrix3D(m_dim_x, m_dim_y, m_dim_z);
    }
    /*
    Reshape matrix without altering underlying memory layout
    */
    void view(int new_x, int new_y, int new_z) {
        size_t new_size = static_cast<size_t>(new_x) * new_y * new_z;
        assert(size() == new_size);
        m_dim_x = new_x;
        m_dim_y = new_y;
        m_dim_z = new_z;
    }
    
    size_t size() const {
        return static_cast<size_t>(m_dim_x) * m_dim_y * m_dim_z;
    }

    Matrix3D repeat(int dim, int times) const {
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

        Matrix3D result(new_dim_x, new_dim_y, new_dim_z);

        // Optimize based on which dimension we're repeating
        if (dim == 0) {
            // Repeat along X dimension (most contiguous in memory)
            const int block_size = m_dim_y * m_dim_z;
            for (int orig_x = 0; orig_x < m_dim_x; ++orig_x) {
                const T* src_block = &(*this)(orig_x, 0, 0);
                for (int t = 0; t < times; ++t) {
                    T* dest_block = &result(orig_x * times + t, 0, 0);
                    std::copy(src_block, src_block + block_size, dest_block);
                }
            }
        }
        else if (dim == 1) {
            // Repeat along Y dimension
            for (int x = 0; x < m_dim_x; ++x) {
                for (int orig_y = 0; orig_y < m_dim_y; ++orig_y) {
                    for (int z = 0; z < m_dim_z; ++z) {
                        const T val = (*this)(x, orig_y, z);
                        for (int t = 0; t < times; ++t) {
                            result(x, orig_y * times + t, z) = val;
                        }
                    }
                }
            }
        }
        else { // dim == 2
            // Repeat along Z dimension
            for (int x = 0; x < m_dim_x; ++x) {
                for (int y = 0; y < m_dim_y; ++y) {
                    const T* src_row = &(*this)(x, y, 0);
                    for (int orig_z = 0; orig_z < m_dim_z; ++orig_z) {
                        const T val = src_row[orig_z];
                        T* dest = &result(x, y, orig_z * times);
                        std::fill(dest, dest + times, val);
                    }
                }
            }
        }

        return result;
    }

    T& operator()(int x, int y, int z) {
        check_indices(x, y, z);
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    const T& operator()(int x, int y, int z) const {
        check_indices(x, y, z);
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    bool operator==(const Matrix3D<T>& other) const {
        if (m_dim_x != other.m_dim_x || m_dim_y != other.m_dim_y || m_dim_z != other.m_dim_z) {
            return false;
        }

        for (int i = 0; i < length(); ++i) {
            if (m_data[i] != other.m_data[i]) {
                IF_DEBUG(
                    std::cout << "value mismatch at index " << i 
                              << ": " << m_data[i] << " != " << other.m_data[i] << std::endl;
                )
                return false;
            }
        }
        return true;
    }

    size_t length() const { return (size_t) m_dim_x * m_dim_y * m_dim_z; }
    
    T sum() const {
        T sum = 0;
        for (int i = 0; i < length(); i++) {
            sum += m_data[i];
        }
        return sum;
    }
    
    T mean() const {
        return length() > 0 ? sum() / length() : T{};
    }
    
    T max() const {
        if (length() == 0) return T{};
        T max_val = m_data[0];
        for (int i = 1; i < length(); i++) {
            if (m_data[i] > max_val) {
                max_val = m_data[i];
            }
        }
        return max_val;
    }
    
    T min() const {
        if (length() == 0) return T{};
        T min_val = m_data[0];
        for (int i = 1; i < length(); i++) {
            if (m_data[i] < min_val) {
                min_val = m_data[i];
            }
        }
        return min_val;
    }
    
    void statistics() const {
        std::cout << "max: " << max() << ", min: " << min() 
                  << ", mean: " << mean() << ", sum: " << sum() << std::endl; 
    }

    T sum(int size) const {
        T sum = 0;
        for (int i = 0; i < size; i++) {
            sum += m_data[i];
        }
        return sum;
    }

    T sum(int size, int start_idx) const {
        T sum = 0;
        for (int i = 0; i < size; i++) {
            sum += m_data[start_idx + i];
        }
        return sum;
    }

    void load(const char* path) 
    {
        read_to_array(path, m_data.get(), length() );
    }
    
    void save(const char* path)
    {
       write_array_to_file(path, m_data.get(), length() ) ;
    }
    
    bool compare_with_gt(const std::string path)
    {
        Matrix3D gt = as_shape();
        gt.load(path.c_str());
        return (*this == gt);
    }

    // Raw data access (use with caution)
    T* data() { return m_data.get(); }
    const T* data() const { return m_data.get(); }
    
    void show(bool verbose = false)
    {
        printf("dim: [%d x %d x %d]\n", m_dim_x, m_dim_y, m_dim_z);
        if (verbose)
        {
            for (int x = 0; x < m_dim_x; ++x) {
                for (int y = 0; y < m_dim_y; ++y) {
                    for (int z = 0; z < m_dim_z; ++z) {
                        std::cout << "Element (" << x << ", " << y << ", " << z << "): " 
                            << (*this)(x, y, z) << std::endl;
                    }
                }
            }
        }
    }

private:
    void check_indices(int x, int y, int z) const {
        if (x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
            throw std::out_of_range("Matrix3D: Indices (" + 
                                  std::to_string(x) + ", " + 
                                  std::to_string(y) + ", " + 
                                  std::to_string(z) + ") out of range for dimensions (" + 
                                  std::to_string(m_dim_x) + ", " + 
                                  std::to_string(m_dim_y) + ", " + 
                                  std::to_string(m_dim_z) + ")");
        }
    }
};

static inline void debug_info(std::string s) {
#ifdef DEBUG
    std::cout << s << std::endl;
#endif
}
#endif
