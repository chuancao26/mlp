#include "../include/SoftMaxCELayer.hpp"
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void softmax_kernel(const float *d_X, float *d_Z, float *prob,
                               int classes, int batch)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch)
    {
        float max_val = d_X[row * classes];
        for (int c = 0; c < classes; c++)
        {
            if (d_X[row * classes + c] > max_val)
            {
                max_val = d_X[row * classes + c];
            }
        }
        float sum = 0.0f;
        // hallamos las prob de cada clase por batchj
        for (int c = 0; c < classes; c++)
        {
            floa e = std::expf(d_X[row * classes + c] - max_val);
            prob[row * classes + c] = e;
            sum += e;
        }

        // dividimos la suma de los e para hallar la probabilidad 
        for (int c = 0; c < classes; c++)
        {
            prob[row * classes + c] /= sum;
            d_Z[row * classes + c] = prob[row * classes + c];
        }
    }
}
__global__ void softmax_backward_kernel(const float* y_true, const float* y_hat,
    float* d_dX, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        d_dX[idx] = y_hat[idx] - y_true[idx];
    }
}

SoftMaxCELayer::SoftMaxCELayer(int n_class, int batch): classes(n_class), max_batch(batch)
{
    cudaMalloc(&d_prob, classes * max_batch * sizeof(float));
    cudaMalloc(&d_total_loss, sizeof(float));
}
SoftMaxCELayer::~SoftMaxCELayer()
{
    cudaFree(d_prob);
    cudaFree(d_total_loss);
}

void SoftMaxCELayer::forward(const float *d_X, float *d_Z, int batch)
{
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(d_X, d_Z, d_prob, classes, batch);
}
void SoftMaxCELayer::backward(const float* d_dOut, float *d_dX, int batch)
{
    int threads = 256;
    int total = classes * batch;
    int blocks = (total + threads -1 ) / threads;
    softmax_backward_kernel<<<blocks, threads>>>(d_dOut, d_prob, d_dX, total);
}
void SoftMaxCELayer::update(float lr)
{
}