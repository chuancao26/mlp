#include "../include/ReLULayer.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <device_launch_parameters.h>


__global__ void forward_kernel(const float* d_X, float* d_Z, float* X_input, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        X_input[idx] = d_X[idx];
        d_Z[idx] = (d_X[idx] > 0) ? d_X[idx] : 0.0;
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
ReLULayer::ReLULayer(int feats, int batch): max_batch(batch), features(feats)
{
    cudaMalloc(&d_X_input, batch * features * sizeof(float));
}
ReLULayer::~ReLULayer()
{
    cudaFree(&d_X_input);

}
void ReLULayer::forward(const float* d_X, float* d_Z, int current_batch)
{
    int total = current_batch * features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    forward_kernel<<<blocks, threads>>>(d_X, d_Z, d_X_input, total);
}
void ReLULayer::backward(const float* d_dOut, float* d_dX, int batch)
{
    int total = batch * features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
}