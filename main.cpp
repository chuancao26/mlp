#include <iostream>
#include <exception>
#include <string>

#include "include/CUDADataset.hpp"
#include "include/MLP.hpp"
#include "include/DenseLayer.hpp"
#include "include/ReLULayer.hpp"
#include "include/SoftMaxCELayer.hpp"
#include "include/Trainer.hpp"

int main() 
{
    // 1. Hiperparámetros base
    const int classes = 10;
    const int batch_size = 256;
    const int epochs = 50;
    const float learning_rate = 0.01f;

    // Rutas a los archivos de MNIST
    // (Asegúrate de que los nombres coincidan exactamente con tus archivos descomprimidos)
    const std::string train_img_path = "mnist_data/train-images-idx3-ubyte";
    const std::string train_lbl_path = "mnist_data/train-labels-idx1-ubyte";
    const std::string test_img_path  = "mnist_data/t10k-images-idx3-ubyte";
    const std::string test_lbl_path  = "mnist_data/t10k-labels-idx1-ubyte";

    std::cout << "=== Preparando Dataset MNIST ===" << std::endl;

    // 2. Instanciamos los cargadores para Train y Test
    CUDADataset train_dataset(classes);
    CUDADataset test_dataset(classes);

    try 
    {
        // Cargamos los datos reales directo a la VRAM
        train_dataset.load_mnist(train_img_path, train_lbl_path);
        test_dataset.load_mnist(test_img_path, test_lbl_path);
    } 
    catch (const std::exception &e) 
    {
        std::cerr << "\n[!] Error crítico cargando datos:\n" << e.what() << std::endl;
        std::cerr << "Verifica que la carpeta 'mnist_data' exista y contenga los archivos." << std::endl;
        return 1;
    }

    // Extraemos las dimensiones reales directamente del dataset
    int features = train_dataset.get_features(); 

    std::cout << "\n[Datos en GPU listos]" << std::endl;
    std::cout << "Entrenamiento -> Muestras: " << train_dataset.get_samples() << " | Features: " << features << std::endl;
    std::cout << "Testeo        -> Muestras: " << test_dataset.get_samples() << " | Features: " << features << std::endl;

    // 3. Construir el Modelo
    std::cout << "\n=== Construyendo Modelo ===" << std::endl;
    MLP model;
    model.add(new DenseLayer(features, 512, batch_size));
    model.add(new ReLULayer(512, batch_size));
    model.add(new DenseLayer(512, 256, batch_size));
    model.add(new ReLULayer(256, batch_size));
    model.add(new DenseLayer(256, classes, batch_size));
    
    model.set_loss_layer(new SoftMaxCELayer(classes, batch_size));

    // 4. Entrenar y Evaluar
    Trainer trainer(&model);
    
    // Ejecutamos el entrenamiento con el dataset de train
    trainer.fit(train_dataset, epochs, batch_size, learning_rate);

    // Ejecutamos la evaluación con el dataset de test (imprimirá el placeholder por ahora)
    float accuracy = trainer.evaluate(test_dataset, batch_size);

    return 0;
}