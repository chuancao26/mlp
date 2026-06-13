#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

#include "Layer.hpp"

class ReLULayer : public Layer
{
private:
    int features, max_batch;
    float* d_X_input;

public:
    ReLULayer(int feats, int batch);
    ~ReLULayer() override;

    void forward(const float *X_input, int batch) override;
    void backward(const float *dOut, int batch) override;
    
    // Ya no necesitas declarar update() si la clase Layer tiene una implementación por defecto vacía.
};

#endif