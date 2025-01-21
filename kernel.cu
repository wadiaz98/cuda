#include <iostream>
// Al ponerle global se hace un kernel de CUDA y debe estar aqui en el .cu
// al usar device se invoca desde el device, el host no tiene acceso y sirve para funciones auxiliares
__global__
void suma_kernel(float* a, float* b, float* c, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x ;

    if(index < size)
    c[index] = a[index] + b[index];
}

extern "C"
void suma_paralela(float* a, float* b, float* c, int size)
{
    int thread_num = 1024;
    int block_num = std::ceil(size/(float)thread_num);

    std::printf("num_blocks: %d, num_threads: %d\n", block_num, thread_num);

    suma_kernel<<<block_num, thread_num>>>(a,b,c,size);
}