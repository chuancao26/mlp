#ifndef LAYER_HPP
#define LAYER_HPP

class Layer
{
public:
    float* d_output = nullptr;     
    float* d_grad_input = nullptr; 

    virtual ~Layer() = default;
    
    virtual void forward(const float *X_input, int batch) = 0;
    
    virtual void backward(const float *dOut_from_next_layer, int batch) = 0;
    
    virtual void update(float lr) {} 
};

#endif