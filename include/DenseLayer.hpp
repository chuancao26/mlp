#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP
#include "Layer.hpp"
#include <cublas_v2.h>

class DenseLayer : public Layer
{
private:
    int in_feats, out_feats, max_batch;
    cublasHandle_t handle;
    float *d_X_input, *d_dW, *d_db, *h_b, *h_W;

public:
    float *d_b, *d_W;

    DenseLayer(int in_f,
               int out_f,
               int batch_size) : in_feats(in_f), out_feats(out_f), max_batch(batch_size);
    ~DenseLayer() override;
    void forward(const float *X_input,
                 float *Z) override;
    void backward(const float *dOut,
                  float *dX) override;
    void update(float lr) override;
};

#endif