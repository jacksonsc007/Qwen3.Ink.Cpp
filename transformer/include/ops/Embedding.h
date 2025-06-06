#include "common.h"
#include <cassert>



class Embedding {
   public:
    Embedding(int embed_dim_, int voc_size_)
        : embed_dim(embed_dim_), voc_size(voc_size_){
            lookup = Matrix3D<float>(1, voc_size_, embed_dim_);
        }
    Embedding(){};
    Matrix3D<float> forward(Matrix3D<int> input_id);
    void load(std::string path);
    int embed_dim, voc_size;
    Matrix3D<float> lookup;
private:
    std::string profile_name = "Embedding";
};


void load_Embedding_params(Embedding &op, std::string prefix);