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

__global__ void softmax_backward_kernel(const float *y_true, const float *y_hat, float *d_dX, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        // Derivada de Softmax + CrossEntropy
        d_dX[idx] = y_hat[idx] - y_true[idx];
    }
}

__global__ void compute_ce_loss_kernel(const float *y_true, const float *y_hat, float *d_loss, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_loss = 0.0f;
    if (idx < total)
    {
        if (y_true[idx] > 0.5f)
        {
            local_loss = -logf(fmaxf(y_hat[idx], 1e-7f));
            atomicAdd(d_loss, local_loss);
        }
    }
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

    softmax_kernel<<<blocks, threads>>>(d_X, d_output, d_prob, classes, batch);
}

void SoftMaxCELayer::backward(const float *y_true, int batch)
{
    int threads = 256;
    int total = classes * batch;
    int blocks = (total + threads - 1) / threads;

    softmax_backward_kernel<<<blocks, threads>>>(y_true, d_prob, d_grad_input, total);
}

float SoftMaxCELayer::compute_loss(const float *y_true, int batch)
{
    float h_zero = 0.0f;
    cudaMemcpy(d_total_loss, &h_zero, sizeof(float), cudaMemcpyHostToDevice);

    int total = classes * batch;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    compute_ce_loss_kernel<<<blocks, threads>>>(y_true, d_output, d_total_loss, total);

    float h_total_loss = 0.0f;
    cudaMemcpy(&h_total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToDevice);

    return h_total_loss / static_cast<float>(batch);
}