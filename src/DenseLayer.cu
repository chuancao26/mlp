#include "../include/DenseLayer.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <device_launch_parameters.h>
#include <cmath> // Para std::sqrt

__global__ void copyCuda(float *dest, const float *src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dest[idx] = src[idx];
    }
}

__global__ void addbiasKernel(float *d_Z, float *d_b, int batch_size, int feats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * feats)
    {
        int col = idx % feats; // Revisa cómo está tu matriz transpuesta (cuida los índices de filas y columnas)
        d_Z[idx] += d_b[col];
    }
}

__global__ void compute_db_kernel(const float *d_Out, float *d_db, int batch_size, int out_feat)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_feat)
    {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++)
        {
            sum += d_Out[i * out_feat + idx];
        }
        d_db[idx] = sum / static_cast<float>(batch_size);
    }
}

__global__ void update_weights(float *d_W, float *d_b,
                               const float *d_dW, const float *d_db, 
                               const float lr, const int in_feat, int out_feat)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_feat * in_feat)
    {
        d_W[idx] -= d_dW[idx] * lr;
    }
    // Usamos el mismo grid, validamos si es válido para los bias
    if (idx < out_feat) 
    {
        d_b[idx] -= d_db[idx] * lr;
    }
}

DenseLayer::DenseLayer(int in_f, int out_f, int batch_size) 
    : in_feats(in_f), out_feats(out_f), max_batch(batch_size)
{
    cublasCreate(&handle);

    // Memoria para la matemática
    cudaMalloc((void **)&d_W, out_feats * in_feats * sizeof(float));
    cudaMalloc((void **)&d_b, out_feats * sizeof(float));
    cudaMalloc((void **)&d_dW, out_feats * in_feats * sizeof(float));
    cudaMalloc((void **)&d_db, out_feats * sizeof(float));
    cudaMalloc((void **)&d_X_input, max_batch * in_feats * sizeof(float));

    // NUEVO: Memoria de salida y gradientes de entrada (Heredados de Layer)
    cudaMalloc((void **)&d_output, max_batch * out_feats * sizeof(float));
    cudaMalloc((void **)&d_grad_input, max_batch * in_feats * sizeof(float));

    // Inicialización Xavier/He
    float *h_W = new float[out_feats * in_feats];
    float *h_b = new float[out_feats];

    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(2.0f / in_feats);
    std::normal_distribution<float> dist(0.0f, limit);

    for (int i = 0; i < in_feats * out_feats; i++)
        h_W[i] = dist(gen);
    for (int i = 0; i < out_feats; i++)
        h_b[i] = 0.0f;

    cudaMemcpy(d_W, h_W, out_feats * in_feats * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, out_feats * sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_b;
    delete[] h_W;
}

DenseLayer::~DenseLayer()
{
    cudaFree(d_b);
    cudaFree(d_W);
    cudaFree(d_db);
    cudaFree(d_dW);
    cudaFree(d_X_input);
    
    // NUEVO: Liberar memoria base
    cudaFree(d_output);
    cudaFree(d_grad_input);
    
    cublasDestroy(handle);
}

// NUEVA FIRMA: El parámetro d_Z se quita
void DenseLayer::forward(const float *X_batch, int batch_size)
{
    int total_elements = batch_size * in_feats;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    // Guardar X para el backward
    copyCuda<<<blocks, threads>>>(d_X_input, X_batch, total_elements);

    float alpha = 1.0f, beta = 0.0f;
    // El resultado se guarda en this->d_output
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_feats, batch_size, in_feats,
                &alpha, d_W, in_feats, d_X_input, in_feats,
                &beta, d_output, out_feats);

    int blocksZ = (batch_size * out_feats + threads - 1) / threads;
    addbiasKernel<<<blocksZ, threads>>>(d_output, d_b, batch_size, out_feats);
}

// NUEVA FIRMA: El parámetro d_dX se quita
void DenseLayer::backward(const float *d_dout, int batch_size)
{
    float alpha_dw = 1.0f / batch_size, beta = 0.0f;

    // Calcular dW
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                in_feats, out_feats, batch_size, 
                &alpha_dw, d_X_input, in_feats, d_dout, out_feats, 
                &beta, d_dW, in_feats);

    // Calcular db
    int threads = 256;
    int blocks_db = (out_feats + threads - 1) / threads;
    compute_db_kernel<<<blocks_db, threads>>>(d_dout, d_db, batch_size, out_feats);

    // Calcular dX (d_grad_input) para pasar a la capa anterior
    float alpha_dx = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                in_feats, batch_size, out_feats, 
                &alpha_dx, d_W, in_feats, d_dout, out_feats, 
                &beta, d_grad_input, in_feats);
}

void DenseLayer::update(float lr)
{
    int threads = 256;
    int blocks = (in_feats * out_feats + threads - 1) / threads;
    update_weights<<<blocks, threads>>>(d_W, d_b, d_dW, d_db, lr, in_feats, out_feats);
}