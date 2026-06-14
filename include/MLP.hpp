#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include "Layer.hpp"
#include "SoftMaxCELayer.hpp"


class MLP
{
    private:
    std::vector<Layer*> layers;
    SoftMaxCELayer* loss_layer;

    public:
    MLP();
    ~MLP();

    void add(Layer* layer);
    void set_loss_layer(SoftMaxCELayer* loss);

    float train_step(const float* d_X_batch, const float* d_y_true, int batch, float lr);
    void predict(const float* d_X_batch, float* d_y_hat, int batch);

};

#endif