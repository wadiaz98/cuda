//
// Created by fing.labcom on 21/01/2025.
//

#include <iostream>
#include <cuda_runtime.h>
const size_t VECTOR_SIZE =  1024*1024;

void suma_serial(float* x, float* y, float* z, int size)
{
    for(int i = 0 ; i < size ; i++)
    {
        z[i] = x[i] + y[i];
    }
}
extern "C"
void suma_paralela(float* a, float* b, float* c, int size);

int main()
{
// (1,1,1....,1) + (2,2,2....,2) = (3,3,3....,3)

    //host
    float* h_A = new float[VECTOR_SIZE];
    float* h_B = new float[VECTOR_SIZE];
    float* h_C = new float[VECTOR_SIZE];

    //Para no hacer el for y llenar
    // todo el bloque de memoria con 0

    memset(h_C, 0, VECTOR_SIZE * sizeof(float));

    for(int i = 0 ; i < VECTOR_SIZE ; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    //device
    float* d_A;
    float* d_B;
    float* d_C;

    //calcular el bloque de memoria a utilizar en bytes

    size_t size_in_bytes = VECTOR_SIZE * sizeof(float);

    cudaMalloc(&d_A, size_in_bytes);
    cudaMalloc(&d_B, size_in_bytes);
    cudaMalloc(&d_C, size_in_bytes);

    //copiar de host a device HTD
    // a donde, de donde y la direccion de memoria
    cudaMemcpy(d_A, h_A, size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_in_bytes, cudaMemcpyHostToDevice);

    // Invocar el kernel desde el archivo cu
    suma_paralela(d_A, d_B, d_C, VECTOR_SIZE);
    // copiar de device a host DTH
    cudaMemcpy(h_C, d_C, size_in_bytes, cudaMemcpyDeviceToHost);

    //imprimer resultados
    std::printf("\n\nresultado: ");
    for(int i= 0; i<10; i++)
    {
        std::printf(" %.0f", h_C[i]);
    }

    // Liberar memoria
    return 0;
}
