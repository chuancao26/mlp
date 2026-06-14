#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "MLP.hpp"
#include "CUDADataset.hpp"

class Trainer
{
private:
    MLP* model;

public:
    Trainer(MLP* model_ptr);
    ~Trainer() = default;

    // Métodos estándar de los frameworks de Deep Learning
    void fit(const CUDADataset& train_data, int epochs, int batch_size, float learning_rate);
    float evaluate(const CUDADataset& test_data, int batch_size);
};

#endif