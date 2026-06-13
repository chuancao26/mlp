#ifndef LAYER_HPP
#define LAYER_HPP

class Layer
{
public:
    // Variables públicas (o protected con getters) para que el modelo principal las lea
    float* d_output = nullptr;     // Reemplaza a Z
    float* d_grad_input = nullptr; // Reemplaza a dX

    virtual ~Layer() = default;
    
    // El resultado se guardará internamente en d_output
    virtual void forward(const float *X_input, int batch) = 0;
    
    // El resultado se guardará internamente en d_grad_input
    virtual void backward(const float *dOut_from_next_layer, int batch) = 0;
    
    // Ponemos llaves {} para no obligar a ReLU a implementar esto vacío
    virtual void update(float lr) {} 
};

#endif