#include "../include/DenseLayer.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <device_launch_parameters.h>

__global__ void copyCuda(float *dest, const float *src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dest[idx] = src[idx];
    }
}
__global__ void addbiasKernel(float* d_Z, float* d_b, int batch_size, int feats)
{
    int idx = blockIdx.x + blockDim.x + threadIdx.x;
    if (idx < batch_size * feats)
    {
        int col = idx % feats;
        d_Z[idx] = d_b[col];
    }
}

DenseLayer::DenseLayer(
    int in_f,
    int out_f,
    int batch_size) : in_feats(in_f),
                      out_feats(out_f),
                      max_batch(batch_size)
{
    cublasCreate(&handle);
    cudaMalloc((void **)&d_W, out_feats * in_feats * sizeof(float));
    cudaMalloc((void **)&d_b, out_feats * sizeof(float));

    cudaMalloc((void **)&d_dW, out_feats * in_feats * sizeof(float));
    cudaMalloc((void **)&d_db, out_feats * sizeof(float));

    cudaMalloc((void **)&d_X_input, max_batch * in_feats * sizeof(float));

    float *h_W = new float[out_feats * in_feats];
    float *h_b = new float[out_feats];

    for (int i = 0; i < out_feats * in_feats; i++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        float limit = std::sqrt(2.0f / in_feats);
        std::normal_distribution<float> dist(0.0f, limit);
        for (int i = 0; i < in_feats * out_feats; i++)
            h_W[i] = dist(gen);
        for (int i = 0; i < out_feats; i++)
            h_b[i] = 0.0f;
        cudaMemcpy(d_W, h_W, out_feats * in_feats * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(h_b, d_b, out_feats * sizeof(float), cudaMemcpyHostToDevice);

        delete[] h_b;
        delete[] h_W;
    }
}
DenseLayer::~DenseLayer()
{
    cudaFree(d_b);
    cudaFree(d_W);
    cudaFree(d_db);
    cudaFree(d_dW);
    cudaFree(d_X_input);
}
void DenseLayer::forward(const float *X_batch,
                         float *d_Z)
{
    // necesitamos hacer multiplicacion de pesos con X y sumar bias
    int total_elements = max_batch * in_feats;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    copyCuda<<<blocks, threads>>>(d_X_input, X_batch, total_elements);

    // multiplicacion de matrices Xinput * W^T

    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_feats, max_batch, in_feats,
                &alpha, d_W, in_feats, d_X_input, in_feats,
                &beta, d_Z, out_feats);

    int blocksZ = (max_batch * out_feats + threads - 1) / threads;
    addbiasKernel<<<blocksZ, threads>>>(d_Z, d_b, 
    max_batch, out_feats);
}
void DenseLayer::backward(const float* d_dout, float* d_dX)
{
    dw = 256 x 784
    X = 64 x 784
    dout = 64 * 256
    dout * X^t
    64* 256 , 64x784

    // necesitamos hacer 2 multplicaciones de matrices i
    // Hallemos el dW y db. dW = d_out * X
    float alpha_dw = 1.0f / max_batch, beta =0.0f;
    cublasSgemm(handle, CUBLAS)

}