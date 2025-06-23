#ifndef COMMON_H
#define COMMON_H
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "utils.h"



template <typename T>
class Matrix3D {
   private:
   public:
    std::unique_ptr<T[]> m_data;
    int m_dim_x, m_dim_y, m_dim_z;

    // Default constructor - creates empty matrix
    Matrix3D() : m_data(nullptr), m_dim_x(0), m_dim_y(0), m_dim_z(0) {}

    // Constructor with dimensions - allocates memory
    Matrix3D(int dim_x, int dim_y, int dim_z) : m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z) {
        IF_DEBUG(printf("\e[31m[INFO]\e[m Constructor with dimensions\n"););
        assert(m_dim_x >= 0 && m_dim_y >= 0 && m_dim_z >= 0);
        if (size() > 0) {
            m_data = std::make_unique<T[]>(size());
        }
    }

    // Option 1: Safe version that always copies data
    Matrix3D(const T* data, int dim_x, int dim_y, int dim_z) : Matrix3D(dim_x, dim_y, dim_z) {
        IF_DEBUG(printf("\e[31m[INFO]\e[m Constructor with dimension and copy from data\n"););
        if (data && size() > 0) {
            std::copy(data, data + size(), m_data.get());
        }
    }

    // Copy constructor
    Matrix3D(const Matrix3D<T>& other) : m_dim_x(other.m_dim_x), m_dim_y(other.m_dim_y), m_dim_z(other.m_dim_z) {
        IF_DEBUG(printf("\e[31m[INFO]\e[m Copy Constructor\n"););
        // printf("\e[31m[INFO]\e[m Copy Constructor\n");
        m_data = std::make_unique<T[]>(other.size());
        std::copy(other.data(), other.data() + other.size(), m_data.get());
    }

    // Move constructor
    Matrix3D(Matrix3D<T>&& other) noexcept
        : m_data(std::move(other.m_data)), m_dim_x(other.m_dim_x), m_dim_y(other.m_dim_y), m_dim_z(other.m_dim_z) {
        IF_DEBUG(printf("\e[31m[INFO]\e[m Move Constructor\n"););
        other.m_dim_x = 0;
        other.m_dim_y = 0;
        other.m_dim_z = 0;
    }

    // Copy assignment
    Matrix3D& operator=(const Matrix3D<T>& other) {
        IF_DEBUG(printf("\e[31m[INFO]\e[m Copy Assignment\n"););
        // printf("\e[31m[INFO]\e[m Copy Assignment\n");
        if (this == &other) return *this;
        // Create temporary and swap
        Matrix3D temp(other);
        swap(*this, temp);
        return *this;
    }

    // Move assignment
    Matrix3D& operator=(Matrix3D<T>&& other) noexcept {
        IF_DEBUG(printf("\e[31m[INFO]\e[m Move Assignment\n"););
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
        IF_DEBUG(printf("\e[31m[INFO]\e[m Swap Function\n"););
        using std::swap;
        swap(first.m_data, second.m_data);
        swap(first.m_dim_x, second.m_dim_x);
        swap(first.m_dim_y, second.m_dim_y);
        swap(first.m_dim_z, second.m_dim_z);
    }

    // concatenate two matrix along x dim
    Matrix3D cat(Matrix3D other) {
        assert(other.m_dim_y == m_dim_y);
        assert(other.m_dim_z == m_dim_z);
        Matrix3D output = Matrix3D(other.m_dim_x + m_dim_x, m_dim_y, m_dim_z);
        std::copy(m_data.get(), m_data.get() + size(), output.data());
        std::copy(other.data(), other.data() + other.size(), output.data() + size());
        return output;
    }

    // exchange dimension 0 and 1
    Matrix3D permute01() {
        PROFILE_START("permute");
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

                std::copy(src_ptr + src_offset, src_ptr + src_offset + dim_z, dest_ptr + dest_offset);
            }
        }

        PROFILE_END("permute");
        return after;
    }

    Matrix3D permute120() {
        PROFILE_START("QwenAttention::permute120");

        const int dim_x = m_dim_y;  // New dimension 0 is old dimension 1
        const int dim_y = m_dim_z;  // New dimension 1 is old dimension 2
        const int dim_z = m_dim_x;  // New dimension 2 is old dimension 0
        Matrix3D after(dim_x, dim_y, dim_z);

        T* dest_ptr = after.m_data.get();
        const T* src_ptr = this->m_data.get();

        const int stride_y = m_dim_z * m_dim_x;
        const int stride_z = m_dim_x;
        const int stride_x = 1;

        for (int i = 0; i < m_dim_x; ++i) {          // Old dim 0 -> new dim 2
            for (int j = 0; j < m_dim_y; ++j) {      // Old dim 1 -> new dim 0
                for (int k = 0; k < m_dim_z; ++k) {  // Old dim 2 -> new dim 1
                    int old_index = i * stride_x + j * stride_y + k * stride_z;
                    int new_index = j * (m_dim_z * m_dim_x) + k * m_dim_x + i;
                    dest_ptr[new_index] = src_ptr[old_index];
                }
            }
        }

        PROFILE_END("QwenAttention::permute120");
        return after;
    }
    /*
    Creates a new Matrix3D with the same shape (dimensions) as another matrix
    The data is uninitialized unless T has a default constructor
    */
    Matrix3D as_shape(const Matrix3D<T>& other) const { return Matrix3D(other.m_dim_x, other.m_dim_y, other.m_dim_z); }

    /*
    Creates a new Matrix3D with the same shape (dimensions) as this matrix
    The data is uninitialized unless T has a default constructor
    */
    Matrix3D as_shape() const { return Matrix3D(m_dim_x, m_dim_y, m_dim_z); }
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

    size_t size() const { return static_cast<size_t>(m_dim_x) * m_dim_y * m_dim_z; }

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
        } else if (dim == 1) {
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
        } else {  // dim == 2
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
        // check_indices(x, y, z);
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    const T& operator()(int x, int y, int z) const {
        // check_indices(x, y, z);
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    bool operator==(const Matrix3D<T>& other) const {
        if (m_dim_x != other.m_dim_x || m_dim_y != other.m_dim_y || m_dim_z != other.m_dim_z) {
            return false;
        }

        for (int i = 0; i < length(); ++i) {
            if (m_data[i] != other.m_data[i]) {
                IF_DEBUG(std::cout << "value mismatch at index " << i << ": " << m_data[i] << " != " << other.m_data[i]
                                   << std::endl;)
                return false;
            }
        }
        return true;
    }

    size_t length() const { return (size_t)m_dim_x * m_dim_y * m_dim_z; }

    T sum() const {
        T sum = 0;
        for (int i = 0; i < length(); i++) {
            sum += m_data[i];
        }
        return sum;
    }

    T mean() const { return length() > 0 ? sum() / length() : T{}; }

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
        std::cout << "max: " << max() << ", min: " << min() << ", mean: " << mean() << ", sum: " << sum() << std::endl;
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

    void load(const char* path) { read_to_array(path, m_data.get(), length()); }

    void save(const char* path) { write_array_to_file(path, m_data.get(), length()); }

    // Raw data access (use with caution)
    T* data() { return m_data.get(); }
    const T* data() const { return m_data.get(); }

    void show(bool verbose = false) {
        printf("dim: [%d x %d x %d]\n", m_dim_x, m_dim_y, m_dim_z);
        if (verbose) {
            for (int x = 0; x < m_dim_x; ++x) {
                for (int y = 0; y < m_dim_y; ++y) {
                    for (int z = 0; z < m_dim_z; ++z) {
                        std::cout << "Element (" << x << ", " << y << ", " << z << "): " << (*this)(x, y, z)
                                  << std::endl;
                    }
                }
            }
        }
    }

    // Extract a sub-tensor along the last dimension: [start_z, end_z)
    Matrix3D<T> slice(int start_z, int end_z) const {
        PROFILE_START("matrix3D slice");
        assert(start_z >= 0 && end_z <= m_dim_z && start_z < end_z);
        Matrix3D<T> result(m_dim_x, m_dim_y, end_z - start_z);
        for (int x = 0; x < m_dim_x; ++x) {
            for (int y = 0; y < m_dim_y; ++y) {
                for (int z = start_z; z < end_z; ++z) {
                    result(x, y, z - start_z) = (*this)(x, y, z);
                }
            }
        }
        PROFILE_END("matrix3D slice");
        return result;
    }

   private:
    void check_indices(int x, int y, int z) const {
        if (x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
            throw std::out_of_range("Matrix3D: Indices (" + std::to_string(x) + ", " + std::to_string(y) + ", " +
                                    std::to_string(z) + ") out of range for dimensions (" + std::to_string(m_dim_x) +
                                    ", " + std::to_string(m_dim_y) + ", " + std::to_string(m_dim_z) + ")");
        }
    }
};

template <typename T>
class MatrixView {
public:
    T* m_data;
    int m_dim_x, m_dim_y, m_dim_z;          // Current dimensions (including repeats)
    int m_dim_x_original, m_dim_y_original, m_dim_z_original; // Original dimensions
    int stride_x, stride_y, stride_z;
    int m_repeat_x, m_repeat_y, m_repeat_z; // Repeat counts per dimension
    int offset_x, offset_y, offset_z;
    MatrixView()
    {
        m_data = NULL;
        m_dim_x = 0;
        m_dim_y = 0;
        m_dim_z = 0;
        m_dim_x_original = 0;
        m_dim_y_original = 0;
        m_dim_z_original = 0;
        stride_x = 0;
        stride_y = 0;
        stride_z = 0;
        m_repeat_x = 1;
        m_repeat_y = 1;
        m_repeat_z = 1;
        offset_x = 0;
        offset_y = 0;
        offset_z = 0;
    }
    // Constructor - wraps existing memory with optional strides
    MatrixView(T* data, int dim_x, int dim_y, int dim_z,
               int stride_x = 0, int stride_y = 0, int stride_z = 0, int offset_x = 0, int offset_y = 0, int offset_z = 0)
        : m_data(data),
          m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z),
          m_dim_x_original(dim_x), m_dim_y_original(dim_y), m_dim_z_original(dim_z),
          stride_x(stride_x > 0 ? stride_x : dim_y * dim_z),
          stride_y(stride_y > 0 ? stride_y : dim_z),
          stride_z(stride_z > 0 ? stride_z : 1),
          offset_x(offset_x), offset_y(offset_y), offset_z(offset_z),
          m_repeat_x(1), m_repeat_y(1), m_repeat_z(1)
    {
        if (!data) {
            throw std::invalid_argument("Data pointer cannot be null");
        }
        if (dim_x <= 0 || dim_y <= 0 || dim_z <= 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
    }
    
    MatrixView(const MatrixView<T> &other)
    {
        m_data = other.m_data;
        m_dim_x = other.m_dim_x;
        m_dim_y = other.m_dim_y;
        m_dim_z = other.m_dim_z;
        m_dim_x_original = other.m_dim_x_original;
        m_dim_y_original = other.m_dim_y_original;
        m_dim_z_original = other.m_dim_z_original;
        stride_x = other.stride_x;
        stride_y = other.stride_y;
        stride_z = other.stride_z;
        m_repeat_x = other.m_repeat_x;
        m_repeat_y = other.m_repeat_y;
        m_repeat_z = other.m_repeat_z;
        offset_x = other.offset_x;
        offset_y = other.offset_y;
        offset_z = other.offset_z;
    }
    
    Matrix3D<T> contiguous()
    {
        Matrix3D<T> output(m_dim_x, m_dim_y, m_dim_z);
        for (int i = 0; i < m_dim_x; i++)
        {
            for (int j = 0; j < m_dim_y; j++)
            {
                for (int k = 0; k < m_dim_z; k++)
                {
                    output(i, j, k) = (*this)(i, j, k);
                }
            }
        }
        return output;
    }

    // Access element with bounds checking and repetition handling
    T& operator()(int x, int y, int z) {
        // modify_repetition_index(x, y, z);
        // check_bounds(x, y, z);
        return m_data[(x + offset_x) * stride_x + (y + offset_y) * stride_y + (z + offset_z) * stride_z];
    }

    const T& operator()(int x, int y, int z) const {
        // modify_repetition_index(x, y, z);
        // check_bounds(x, y, z);
        return m_data[(x + offset_x) * stride_x + (y + offset_y) * stride_y + (z + offset_z) * stride_z];
    }

    // Create a view with a dimension repeated
    MatrixView<T> repeat_dimension(int dim, int times) const {
        if (times <= 0) {
            throw std::invalid_argument("Repeat times must be positive");
        }

        MatrixView<T> new_view = *this;
        switch (dim) {
            case 0: 
                new_view.m_dim_x *= times;
                new_view.m_repeat_x *= times;
                break;
            case 1:
                new_view.m_dim_y *= times;
                new_view.m_repeat_y *= times;
                break;
            case 2:
                new_view.m_dim_z *= times;
                new_view.m_repeat_z *= times;
                break;
            default:
                throw std::invalid_argument("Invalid dimension (0=x, 1=y, 2=z)");
        }
        return new_view;
    }

    void load(const char* path) 
    {
        read_to_array(path, m_data, length() );
    }

    size_t length() const {
        return (size_t)m_dim_x * m_dim_y * m_dim_z;
    }

    // Raw data access
    T* data() { return m_data; }
    const T* data() const { return m_data; }

    // Dimensions
    int dim_x() const { return m_dim_x; }
    int dim_y() const { return m_dim_y; }
    int dim_z() const { return m_dim_z; }

    // Original dimensions (without repetition)
    int original_dim_x() const { return m_dim_x_original; }
    int original_dim_y() const { return m_dim_y_original; }
    int original_dim_z() const { return m_dim_z_original; }


    // Repeat counts
    int repeat_x() const { return m_repeat_x; }
    int repeat_y() const { return m_repeat_y; }
    int repeat_z() const { return m_repeat_z; }

    void modify_repetition_index(int& x, int& y, int& z) const {
        if (m_repeat_x > 1) x = x / m_repeat_x;
        if (m_repeat_y > 1) y = y / m_repeat_y;
        if (m_repeat_z > 1) z = z / m_repeat_z;
    }

    // Copy data to another MatrixView object
    void copy_to(MatrixView<T>& dest) const {
        // Check if dimensions match
        if (dest.m_dim_x != m_dim_x || dest.m_dim_y != m_dim_y || dest.m_dim_z != m_dim_z) {
            throw std::invalid_argument("Destination MatrixView dimensions do not match source dimensions");
        }

        // Copy data element by element
        for (int i = 0; i < m_dim_x; i++) {
            for (int j = 0; j < m_dim_y; j++) {
                for (int k = 0; k < m_dim_z; k++) {
                    dest(i, j, k) = (*this)(i, j, k);
                }
            }
        }
    }
    
    T sum()
    {
        T acc = 0;
        for (int i = 0; i < m_dim_x; i++) {
            for (int j = 0; j < m_dim_y; j++) {
                for (int k = 0; k < m_dim_z; k++) {
                     acc += (*this)(i, j, k);
                }
            }
        }
        return acc;
    }
    
    void statistics()
    {
        std::cout << " sum: " << sum() << std::endl;
    }
};
#endif
