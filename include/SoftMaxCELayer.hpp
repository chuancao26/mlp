#ifndef SOFTMAX_CE_LAYER_HPP
#define SOFTMAX_CE_LAYER_HPP

#include "../include/Layer.hpp"

class SoftMaxCELayer : public Layer
{
private:
    int classes, max_batch;
    float *d_prob, *d_total_loss;

public:
    SoftMaxCELayer(int n_class, int batch);
    ~SoftMaxCELayer() override;

    void forward(const float *X_input, int batch) override;
    
    void backward(const float *y_true, int batch) override;
    
    float compute_loss(const float* y_true, int batch); 

    int get_classes();
};

#endif