--kernel void suma_opencl(__global float* a, __global float* b, __global float* c, const unsigned int size) {
    int index = get_global_id(0);

    if (index < size) {
        c[index] = a[index] + b[index];
    }
}