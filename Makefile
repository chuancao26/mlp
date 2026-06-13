# ==========================================
# Makefile para el MLP Modular en CUDA
# ==========================================

# 1. Compilador
NVCC = nvcc

# 2. Banderas de compilación
# -std=c++14 : Usamos C++14 (necesario para <random>, <chrono>, etc.)
# -O3        : Máxima optimización de rendimiento en CPU/GPU
# -I./include: Le dice al compilador dónde buscar los archivos .hpp
NVCC_FLAGS = -std=c++14 -O3 -I./include

# 3. Librerías
# -lcublas es OBLIGATORIO porque usamos cublasSgemm en DenseLayer
LIBS = -lcublas

# 4. Archivos fuente y objetos
# Definimos dónde están nuestros archivos .cu y .cpp
CU_SRCS = src/DenseLayer.cu src/ReLULayer.cu src/SoftMaxCELayer.cu
CPP_SRCS = main.cpp

# Convertimos la lista de fuentes en una lista de objetos (.o)
CU_OBJS = $(CU_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(CU_OBJS) $(CPP_OBJS)

# 5. Nombre del ejecutable final
TARGET = mlp_cuda

# ==========================================
# Reglas de Compilación
# ==========================================

# Regla por defecto: compilar todo
all: $(TARGET)

# Cómo construir el ejecutable final (Enlazado)
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Cómo compilar los archivos .cu a .o
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Cómo compilar los archivos .cpp a .o
%.o: %.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Regla para limpiar los archivos generados
clean:
	rm -f src/*.o *.o $(TARGET)

# Regla rápida para compilar y ejecutar de un solo golpe
run: $(TARGET)
	./$(TARGET)