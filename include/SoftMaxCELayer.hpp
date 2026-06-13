#ifndef SOFTMAX_CE_LAYER
#define SOFTMAX_CE_LAYER

#include "../include/Layer.hpp"

class SoftMaxCELayer: public Layer
{
    private:
    int classes, max_batch;
    float *d_prob, *d_total_loss;

    public:
    SoftMaxCELayer(int n_class, int batch)
    {
        cudaMalloc(&d_prob, classes * max_batch * sizeof(float));
        cudaMalloc(&d_total_loss, )
    }

    void forward(const float *X_input,
                 float *Z,
                 int batch) override;
    void backward(const float *dOut,
                  float *dX,
                  int batch) override;
    void update(float lr) override;
}









#endif