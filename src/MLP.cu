#include "../include/MLP.hpp"
#include "../include/SoftMaxCELayer.hpp"

MLP::MLP() : loss_layer(nullptr)
{
}
MLP::~MLP()
{
    for (Layer *layer : layers)
    {
        delete layer;
    }
    if (loss_layer)
    {
        delete loss_layer;
    }
}
void MLP::add(Layer *layer)
{
    layers.push_back(layer);
}
void MLP::set_loss_layer(SoftMaxCELayer *loss)
{
    loss_layer = loss;
}
float MLP::train_step(const float *d_X_batch, const float *d_y_true, int batch, float lr)
{
    const float *current_input = d_X_batch;
    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i]->forward(current_input, batch);
        current_input = layers[i]->d_output;
    }
    loss_layer->forward(current_input, batch);

    loss_layer->backward(d_y_true, batch);
    const float *current_grad = loss_layer->d_grad_input;
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        layers[i]->backward(current_grad, batch);
        current_grad = layers[i]->d_grad_input;

        layers[i]->update(lr);
    }
    return loss_layer->compute_loss(d_y_true, batch);
}

void MLP::predict(const float *d_X_batch, float *d_y_hat, int batch)
{
    const float* current_input = d_X_batch;
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->forward(current_input, batch);
        current_input = layers[i]->d_output;
    }
    loss_layer->forward(current_input, batch);

    int total_elements = batch * loss_layer->get_classes();

    cudaMemcpy(d_y_hat, loss_layer->d_output, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

}

