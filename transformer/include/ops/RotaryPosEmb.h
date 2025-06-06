#include "common.h"
#include <cstdlib>
#include "utils.h"

class RotaryPosEmb
{
public:
    RotaryPosEmb(int max_sqlen, int head_dim, std::string path)
    {
        cos = Matrix3D<float>(1, max_sqlen, head_dim);
        cos.load((path + "/cos_cached.bin").c_str());
        sin = Matrix3D<float>(1, max_sqlen, head_dim);
        sin.load((path + "/sin_cached.bin").c_str());
    };
    RotaryPosEmb(){};
    void apply(Matrix3D<float> &key, Matrix3D<float> &value, int start_idx, int len);
    Matrix3D<float> cos, sin;

private:
    std::string profile_name = "RotaryPosEmb";
};

void load_RotaryPosEmb(RotaryPosEmb &op, std::string prefix);