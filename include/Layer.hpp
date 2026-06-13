#ifndef LAYER_HPP
#define LAYER_HPP
class Layer
{
public:
    virtual ~Layer() = default;
    virtual void forward(const float *X,
                         float *Z,
                         int) = 0;
    virtual void backward(const float *dOut,
                          float *d,
                          int) = 0;
    virtual void update(float) = 0;
};

#endif