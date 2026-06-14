#include "../include/Trainer.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

Trainer::Trainer(MLP* model_ptr) : model(model_ptr) {}

void Trainer::fit(const CUDADataset& train_data, int epochs, int batch_size, float learning_rate)
{
    int num_samples = train_data.get_samples();
    int features = train_data.get_features();
    int classes = train_data.get_classes();
    
    const float* d_X_global = train_data.get_X_ptr();
    const float* d_y_global = train_data.get_y_ptr();

    std::cout << "==========================================\n";
    std::cout << "Iniciando Entrenamiento (fit)\n";
    std::cout << "Muestras: " << num_samples << " | Épocas: " << epochs << " | Batch Size: " << batch_size << "\n";
    std::cout << "==========================================\n";

    auto start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int i = 0; i < num_samples; i += batch_size)
        {
            int current_batch = std::min(batch_size, num_samples - i);
            if(current_batch != batch_size) continue; // Por simplicidad, saltamos el lote incompleto

            const float* d_X_batch = d_X_global + (i * features);
            const float* d_y_batch = d_y_global + (i * classes);

            epoch_loss += model->train_step(d_X_batch, d_y_batch, current_batch, learning_rate);
            num_batches++;
        }

        std::cout << "[Epoch " << epoch + 1 << "/" << epochs << "] Loss: " 
                  << (epoch_loss / num_batches) << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Entrenamiento completado en " << elapsed.count() << " segundos.\n";
}

float Trainer::evaluate(const CUDADataset& test_data, int batch_size)
{
    // Aquí implementaremos la métrica de Accuracy pronto usando model->predict()
    std::cout << "Evaluando modelo..." << std::endl;
    return 0.0f; // Placeholder para el accuracy
}