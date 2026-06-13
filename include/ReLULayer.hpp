#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

#include "Layer.hpp"

class ReLULayer:public Layer
{
    private:
    int features, max_batch;
    float* d_X_input;

    public:
    ReLULayer(int feats, int batch);
    ~ReLULayer();
    void forward(const float *X_input,
                 float *Z,
                 int batch) override;
    void backward(const float *dOut,
                  float *dX,
                  int batch) override;
    void update(float lr) override;
    
};

#endif