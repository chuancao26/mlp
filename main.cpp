#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

// Incluimos nuestras capas modulares
#include "include/Layer.hpp"
#include "include/DenseLayer.hpp"
#include "include/ReLULayer.hpp"
#include "include/SoftMaxCELayer.hpp"

// ========================================================================
// 1. EL ORQUESTADOR: SequentialMLP_CUDA
// ========================================================================
class SequentialMLP_CUDA {
private:
    std::vector<Layer*> layers;
    SoftMaxCELayer* loss_layer; // La capa final que calcula el loss

public:
    SequentialMLP_CUDA() {
        loss_layer = nullptr;
    }

    ~SequentialMLP_CUDA() {
        // Liberamos la memoria de todas las capas
        for (auto layer : layers) {
            delete layer;
        }
        if (loss_layer) {
            delete loss_layer;
        }
    }

    // Método para apilar capas dinámicamente
    void add(Layer* layer) {
        layers.push_back(layer);
    }

    void set_loss_layer(SoftMaxCELayer* loss) {
        loss_layer = loss;
    }

    // Un solo paso de entrenamiento
    void train_step(const float* d_X_batch, const float* d_y_true, const int batch_size, float lr) {
        
        // --- FORWARD PASS ---
        const float* current_input = d_X_batch;
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i]->forward(current_input, batch_size);
            // ¡La magia modular! La salida de esta capa es la entrada de la siguiente
            current_input = layers[i]->d_output; 
        }

        // El forward de la capa de Loss (Softmax)
        loss_layer->forward(current_input, batch_size);

        // --- BACKWARD PASS ---
        // Iniciamos la propagación hacia atrás pasando las etiquetas reales
        loss_layer->backward(d_y_true, batch_size);
        
        const float* current_grad = loss_layer->d_grad_input;
        
        // Iteramos hacia atrás a través de las capas ocultas
        for (int i = layers.size() - 1; i >= 0; i--) {
            layers[i]->backward(current_grad, batch_size);
            current_grad = layers[i]->d_grad_input;
            
            // Actualizamos los pesos (ReLU ignorará esto)
            layers[i]->update(lr);
        }
    }
};

// ========================================================================
// 2. FUNCIÓN MAIN
// ========================================================================
int main() {
    // Parámetros del modelo (Simulando MNIST)
    const int num_samples = 60000;
    const int features = 784; // 28x28 pixels
    const int classes = 10;
    const int batch_size = 256;
    const int epochs = 5;
    const float learning_rate = 0.01f;

    std::cout << "Inicializando entorno CUDA y Memoria Host..." << std::endl;

    // 1. Generamos datos ficticios (Data Dummy) para entrenar
    // En la vida real, aquí cargarías las imágenes y labels de MNIST
    float* h_X_train = new float[num_samples * features];
    float* h_y_train = new float[num_samples * classes];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < num_samples * features; i++) {
        h_X_train[i] = dis(gen);
    }
    for (int i = 0; i < num_samples * classes; i++) {
        h_y_train[i] = 0.0f;
    }
    // Asignamos una clase "correcta" arbitraria por simulación (One-Hot)
    for (int i = 0; i < num_samples; i++) {
        int label = i % classes;
        h_y_train[i * classes + label] = 1.0f;
    }

    // 2. Transferencia a la GPU (Un solo viaje global, como en tu código original)
    float *d_X_train_global, *d_y_train_global;
    cudaMalloc(&d_X_train_global, num_samples * features * sizeof(float));
    cudaMalloc(&d_y_train_global, num_samples * classes * sizeof(float));

    cudaMemcpy(d_X_train_global, h_X_train, num_samples * features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_train_global, h_y_train, num_samples * classes * sizeof(float), cudaMemcpyHostToDevice);

    // 3. CONSTRUCCIÓN DEL MODELO MODULAR
    std::cout << "Construyendo modelo (784 -> 512 -> ReLU -> 256 -> ReLU -> 10)" << std::endl;
    SequentialMLP_CUDA model;
    
    // Ahora puedes añadir las capas que quieras, en el orden que quieras
    model.add(new DenseLayer(784, 512, batch_size));
    model.add(new ReLULayer(512, batch_size));
    model.add(new DenseLayer(512, 256, batch_size));
    model.add(new ReLULayer(256, batch_size));
    model.add(new DenseLayer(256, 10, batch_size));
    
    model.set_loss_layer(new SoftMaxCELayer(10, batch_size));

    // 4. BUCLE DE ENTRENAMIENTO
    std::cout << "Iniciando entrenamiento..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Punteros simulados para los batches (Avanzarán en la memoria global)
    for (int epoch = 0; epoch < epochs; epoch++) {
        
        for (int i = 0; i < num_samples; i += batch_size) {
            int current_batch = std::min(batch_size, num_samples - i);
            
            // Si el batch es menor que max_batch (al final del dataset), lo saltamos por simplicidad
            // (Tus capas están dimensionadas para max_batch)
            if(current_batch != batch_size) continue;

            // Desplazamos los punteros para simular la extracción del lote
            const float* d_X_batch = d_X_train_global + (i * features);
            const float* d_y_batch = d_y_train_global + (i * classes);

            // ¡Solo una línea para entrenar todo el lote!
            model.train_step(d_X_batch, d_y_batch, current_batch, learning_rate);
        }
        
        std::cout << "[Epoch " << epoch + 1 << "/" << epochs << "] completada." << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Entrenamiento finalizado en " << elapsed.count() << " segundos." << std::endl;

    // Limpieza
    delete[] h_X_train;
    delete[] h_y_train;
    cudaFree(d_X_train_global);
    cudaFree(d_y_train_global);

    return 0;
}