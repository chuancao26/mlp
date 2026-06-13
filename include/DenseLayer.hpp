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
               int max_batch_size);
    ~DenseLayer() override;
    void forward(const float *X_input,
                 float *Z,
                 int batch) override;
    void backward(const float *dOut,
                  float *dX,
                  int batch) override;
    void update(float lr) override;
};

#endif