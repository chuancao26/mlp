#include "../include/SoftMaxCELayer.hpp"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void softmax_kernel(const float *d_X, float *d_Z, float *prob, int classes, int batch)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch)
    {
        float max_val = d_X[row * classes];
        for (int c = 1; c < classes; c++)
        {
            if (d_X[row * classes + c] > max_val)
            {
                max_val = d_X[row * classes + c];
            }
        }
        float sum = 0.0f;
        for (int c = 0; c < classes; c++)
        {
            float e = expf(d_X[row * classes + c] - max_val);
            prob[row * classes + c] = e;
            sum += e;
        }

        for (int c = 0; c < classes; c++)
        {
            prob[row * classes + c] /= sum;
            d_Z[row * classes + c] = prob[row * classes + c];
        }
    }
}

__global__ void softmax_backward_kernel(const float* y_true, const float* y_hat, float* d_dX, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        // Derivada de Softmax + CrossEntropy
        d_dX[idx] = y_hat[idx] - y_true[idx];
    }
}

// CORRECCIÓN: Un kernel para calcular el Loss si lo necesitas en la CPU
__global__ void compute_ce_loss_kernel(const float* y_true, const float* y_hat, float* d_loss, int classes, int batch)
{
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_loss = 0.0f;
    if (idx < batch * classes) {
        if (y_true[idx] > 0.5f) { // Suponiendo One-Hot encoding
            local_loss = -logf(fmaxf(y_hat[idx], 1e-7f));
        }
    }
    // Aquí requeriría una reducción, pero por simplicidad omitimos la implementación paralela completa
    // del loss en este bloque a menos que tengas tu propia función.
}

SoftMaxCELayer::SoftMaxCELayer(int n_class, int batch) : classes(n_class), max_batch(batch)
{
    cudaMalloc(&d_prob, classes * max_batch * sizeof(float));
    cudaMalloc(&d_total_loss, sizeof(float));
    
    // NUEVO: Memoria base
    cudaMalloc(&d_output, classes * max_batch * sizeof(float));
    cudaMalloc(&d_grad_input, classes * max_batch * sizeof(float));
}

SoftMaxCELayer::~SoftMaxCELayer()
{
    cudaFree(d_prob);
    cudaFree(d_total_loss);
    cudaFree(d_output);
    cudaFree(d_grad_input);
}

void SoftMaxCELayer::forward(const float *d_X, int batch)
{
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    // Guardamos la salida en d_output
    softmax_kernel<<<blocks, threads>>>(d_X, d_output, d_prob, classes, batch);
}

// En la capa final, el 'd_dOut' que recibes son realmente las etiquetas reales (y_true)
void SoftMaxCELayer::backward(const float* y_true, int batch)
{
    int threads = 256;
    int total = classes * batch;
    int blocks = (total + threads - 1) / threads;
    // Calculamos el error inicial (d_grad_input) para la capa anterior
    softmax_backward_kernel<<<blocks, threads>>>(y_true, d_prob, d_grad_input, total);
}


// Método extra que sugerí en el HPP
float SoftMaxCELayer::compute_loss(const float* y_true, int batch) {
    // Si tienes implementada la reducción en GPU, lánzala aquí.
    // De lo contrario, puedes traer un fragmento a CPU para calcularlo (o dejarlo en 0.0f por ahora).
    return 0.0f; 
}