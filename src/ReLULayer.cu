#include "../include/ReLULayer.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void forward_kernel(const float* d_X, float* d_Z, float* X_input, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        X_input[idx] = d_X[idx];
        d_Z[idx] = (d_X[idx] > 0) ? d_X[idx] : 0.0f;
    }
}

__global__ void backward_kernel(const float* d_dOut, float* d_dX, float* d_X_input, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        d_dX[idx] = (d_X_input[idx] > 0) ? d_dOut[idx] : 0.0f;
    }
}

ReLULayer::ReLULayer(int feats, int batch) : max_batch(batch), features(feats)
{
    cudaMalloc(&d_X_input, max_batch * features * sizeof(float));
    
    // NUEVO: Memoria base para forward y backward
    cudaMalloc(&d_output, max_batch * features * sizeof(float));
    cudaMalloc(&d_grad_input, max_batch * features * sizeof(float));
}

ReLULayer::~ReLULayer()
{
    cudaFree(d_X_input);
    cudaFree(d_output);
    cudaFree(d_grad_input);
}

// NUEVA FIRMA
void ReLULayer::forward(const float* d_X, int current_batch)
{
    int total = current_batch * features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Guarda el resultado en d_output
    forward_kernel<<<blocks, threads>>>(d_X, d_output, d_X_input, total);
}

// NUEVA FIRMA
void ReLULayer::backward(const float* d_dOut, int batch)
{
    int total = batch * features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Calcula el gradiente y lo guarda en d_grad_input
    backward_kernel<<<blocks, threads>>>(d_dOut, d_grad_input, d_X_input, total);
}