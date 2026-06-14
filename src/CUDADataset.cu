#include "../include/CUDADataset.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

CUDADataset::CUDADataset(int n_classes) 
    : classes(n_classes), num_samples(0), features(0), d_X(nullptr), d_y(nullptr) 
{
}

CUDADataset::~CUDADataset()
{
    if (d_X) cudaFree(d_X);
    if (d_y) cudaFree(d_y);
}

uint32_t CUDADataset::swap_endian(uint32_t val)
{
    return ((val << 24) & 0xff000000) | 
           ((val << 8)  & 0x00ff0000) | 
           ((val >> 8)  & 0x0000ff00) | 
           ((val >> 24) & 0x000000ff);
}

void CUDADataset::load_mnist(const std::string& images_path, const std::string& labels_path)
{
    std::cout << "-> Cargando imágenes desde: " << images_path << std::endl;
    std::ifstream file_img(images_path, std::ios::binary);
    if (!file_img.is_open()) throw std::runtime_error("No se pudo abrir: " + images_path);

    std::cout << "-> Cargando etiquetas desde: " << labels_path << std::endl;
    std::ifstream file_lbl(labels_path, std::ios::binary);
    if (!file_lbl.is_open()) throw std::runtime_error("No se pudo abrir: " + labels_path);

    // 1. Leer cabeceras de imágenes
    uint32_t magic_img = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file_img.read(reinterpret_cast<char*>(&magic_img), 4);
    file_img.read(reinterpret_cast<char*>(&num_images), 4);
    file_img.read(reinterpret_cast<char*>(&num_rows), 4);
    file_img.read(reinterpret_cast<char*>(&num_cols), 4);
    
    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    // 2. Leer cabeceras de etiquetas
    uint32_t magic_lbl = 0, num_items = 0;
    file_lbl.read(reinterpret_cast<char*>(&magic_lbl), 4);
    file_lbl.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);

    // Validación de integridad
    if (num_images != num_items) {
        throw std::runtime_error("El número de imágenes y etiquetas no coincide.");
    }

    // Actualizamos las dimensiones del dataset
    this->num_samples = num_images;
    this->features = num_rows * num_cols;

    // 3. Crear buffers planos (1D) en CPU 
    // std::vector asegura memoria contigua. Inicializamos y_labels en 0.0f
    std::vector<float> h_X(num_samples * features);
    std::vector<float> h_y(num_samples * classes, 0.0f);

    // 4. Leer y normalizar píxeles (0-255 a 0.0-1.0)
    for (int i = 0; i < num_samples * features; ++i)
    {
        unsigned char pixel = 0;
        file_img.read(reinterpret_cast<char*>(&pixel), 1);
        h_X[i] = static_cast<float>(pixel) / 255.0f;
    }

    // 5. Leer y aplicar One-Hot Encoding a las etiquetas
    for (int i = 0; i < num_samples; ++i)
    {
        unsigned char label = 0;
        file_lbl.read(reinterpret_cast<char*>(&label), 1);
        h_y[i * classes + static_cast<int>(label)] = 1.0f;
    }

    file_img.close();
    file_lbl.close();

    // 6. Asignar memoria en VRAM (GPU) si no se ha asignado antes
    if (d_X) cudaFree(d_X);
    if (d_y) cudaFree(d_y);
    
    cudaMalloc(&d_X, num_samples * features * sizeof(float));
    cudaMalloc(&d_y, num_samples * classes * sizeof(float));

    // 7. Transferencia masiva a la GPU
    std::cout << "-> Transfiriendo " << num_samples << " muestras a la GPU..." << std::endl;
    cudaMemcpy(d_X, h_X.data(), num_samples * features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), num_samples * classes * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "[OK] Dataset cargado y alojado en VRAM exitosamente.\n";
}