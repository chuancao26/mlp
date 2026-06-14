#include "../include/Trainer.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

Trainer::Trainer(MLP *model_ptr) : model(model_ptr) {}

void Trainer::fit(const CUDADataset &train_data, int epochs, int batch_size, float learning_rate)
{
    int num_samples = train_data.get_samples();
    int features = train_data.get_features();
    int classes = train_data.get_classes();

    const float *d_X_global = train_data.get_X_ptr();
    const float *d_y_global = train_data.get_y_ptr();

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
            if (current_batch != batch_size)
                continue; // Por simplicidad, saltamos el lote incompleto

            const float *d_X_batch = d_X_global + (i * features);
            const float *d_y_batch = d_y_global + (i * classes);

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

float Trainer::evaluate(const CUDADataset &test_data, int batch_size)
{
    int num_samples = test_data.get_samples();
    int features = test_data.get_features();
    int classes = test_data.get_classes();

    const float *d_X_global = test_data.get_X_ptr();
    const float *d_y_global = test_data.get_y_ptr();

    std::cout << "\n==========================================\n";
    std::cout << "Evaluando Modelo...\n";

    // 1. Buffers temporales para traer las respuestas de la GPU a la CPU
    float *h_y_pred = new float[batch_size * classes];
    float *h_y_true = new float[batch_size * classes];

    // Buffer en la GPU donde el modelo dejará sus predicciones
    float *d_y_pred;
    cudaMalloc(&d_y_pred, batch_size * classes * sizeof(float));

    int correct_predictions = 0;
    int total_evaluated = 0;

    // 2. Iteramos sobre el dataset de testeo
    for (int i = 0; i < num_samples; i += batch_size)
    {
        int current_batch = std::min(batch_size, num_samples - i);
        if (current_batch != batch_size)
            continue; // Ignorar el último lote incompleto por ahora

        const float *d_X_batch = d_X_global + (i * features);
        const float *d_y_batch = d_y_global + (i * classes);

        // La GPU hace la predicción mágicamente
        model->predict(d_X_batch, d_y_pred, current_batch);

        // Traemos las predicciones y las etiquetas reales a la CPU para compararlas
        cudaMemcpy(h_y_pred, d_y_pred, current_batch * classes * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y_true, d_y_batch, current_batch * classes * sizeof(float), cudaMemcpyDeviceToHost);

        // 3. Lógica para encontrar el valor máximo (argmax)
        for (int j = 0; j < current_batch; j++)
        {
            int pred_class = 0;
            int true_class = 0;
            float max_prob = h_y_pred[j * classes];
            float max_true = h_y_true[j * classes];

            for (int c = 1; c < classes; c++)
            {
                if (h_y_pred[j * classes + c] > max_prob)
                {
                    max_prob = h_y_pred[j * classes + c];
                    pred_class = c;
                }
                if (h_y_true[j * classes + c] > max_true)
                {
                    max_true = h_y_true[j * classes + c];
                    true_class = c;
                }
            }

            // Si la clase que predijo la red es igual a la real, es un acierto
            if (pred_class == true_class)
            {
                correct_predictions++;
            }
            total_evaluated++;
        }
    }

    // 4. Limpieza
    cudaFree(d_y_pred);
    delete[] h_y_pred;
    delete[] h_y_true;

    // 5. Cálculo y resultado
    float accuracy = (static_cast<float>(correct_predictions) / total_evaluated) * 100.0f;
    std::cout << "Accuracy en Test: " << accuracy << "% (" << correct_predictions << "/" << total_evaluated << " correctos)\n";
    std::cout << "==========================================\n";

    return accuracy;
}