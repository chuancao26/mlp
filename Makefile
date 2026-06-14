# ==========================================
# Makefile para el MLP Modular en CUDA
# ==========================================

# 1. Compilador
NVCC = nvcc

# 2. Banderas de compilación
# -std=c++14 : Usamos C++14
# -O3        : Máxima optimización de rendimiento en CPU/GPU
# -I./include: Le dice al compilador dónde buscar los archivos .hpp
# -Wno-deprecated-gpu-targets: Silencia advertencias en GPUs modernas
NVCC_FLAGS = -std=c++14 -O3 -I./include -Wno-deprecated-gpu-targets

# 3. Librerías
LIBS = -lcublas

# 4. Archivos fuente y objetos
# ¡CORREGIDO!: CUDADataset.cu y MLP.cu ahora están en la lista de archivos CUDA
CU_SRCS = src/DenseLayer.cu src/ReLULayer.cu src/SoftMaxCELayer.cu src/CUDADataset.cu src/MLP.cu
# Trainer y main siguen siendo .cpp estándar
CPP_SRCS = main.cpp src/Trainer.cpp

# Convertimos la lista de fuentes en una lista de objetos (.o)
CU_OBJS = $(CU_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(CU_OBJS) $(CPP_OBJS)

# 5. Nombre del ejecutable final
TARGET = mlp_cuda

# ==========================================
# Reglas de Compilación
# ==========================================

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f src/*.o *.o $(TARGET)

run: $(TARGET)
	./$(TARGET)