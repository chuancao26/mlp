#ifndef CUDA_DATASET_HPP
#define CUDA_DATASET_HPP

#include <string>
#include <cstdint>

class CUDADataset
{
private:
    float *d_X;
    float *d_y;
    int num_samples;
    int features;
    int classes;

    // Método utilitario privado para los archivos binarios
    uint32_t swap_endian(uint32_t val);

public:
    CUDADataset(int n_classes);
    ~CUDADataset();

    // El método principal que cargará todo directo a la GPU
    void load_mnist(const std::string& images_path, const std::string& labels_path);

    // Getters
    int get_samples() const { return num_samples; }
    int get_features() const { return features; }
    int get_classes() const { return classes; }
    const float* get_X_ptr() const { return d_X; }
    const float* get_y_ptr() const { return d_y; }
};

#endif