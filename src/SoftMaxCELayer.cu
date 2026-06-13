#include <>
    SoftMaxCELayer(int n_class, int batch)
    {
        cudaMalloc(&d_prob, classes * max_batch * sizeof(float));
        cudaMalloc(&d_total_loss, )
    }