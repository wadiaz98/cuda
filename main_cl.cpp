#include <iostream>
#include <Cl/cl.h>

//codigo no cuda, codigo escrito en OpenCL
std::string kernel_code =
    "__kernel void suma_opencl(__global float* a, __global float* b, __global float* c, const unsigned int size) {"
    "   int index = get_global_id(0);           "
    "   if (index < size) {                     "
    "       c[index] = a[index] + b[index];     "
    "   }"
    "}"
;

int main()
{
    cl_platform_id* platforms = nullptr;
    cl_uint num_platforms = 0;

    //--Obtener numero de plataformas
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::printf("Total de plataformas: %d\n", num_platforms);

    platforms = new cl_platform_id[num_platforms];
    clGetPlatformIDs(num_platforms, platforms, nullptr);
    for(int i = 0; i<num_platforms; i++)
    {
        char vendor[1024];
        char version[1024];
        char name[1024];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 1024, vendor, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 1024, version, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, name, nullptr);
        std::printf("Plataforma %d\n", i);
        std::printf(" Vendor: %s\n", vendor);
        std::printf(" Nombre: %s\n", name);
        std::printf(" Version: %s\n", version);

        std::printf(" --devices\n");
        cl_uint num_devices= 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,0, nullptr, &num_devices);
        std::printf(" Total de dispositivos: %d\n", num_devices);

        cl_device_id* devices = new cl_device_id[num_devices];

        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, nullptr);
        char device_name[1024];

        for( int di = 0; di<num_devices; di++)
        {

            clGetDeviceInfo(devices[di], CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);

            std::printf(" device_%d: name: %s\n", di, device_name);
        }

    }

    //---------------------------------------------
    std::printf("\n");
    std::printf("----------------------------------------\n");

    //Una vez identificadas las plataformas y dispositivos
    //seleccionar la plataforma y el dispositivo

    //Si se usa intel toca asegurarse que se use la GPU
    cl_platform_id platform = platforms[4];
    cl_device_id device_id = nullptr; // device GPU
    cl_context context = nullptr;
    cl_command_queue commands_queue = nullptr;
    cl_program program = nullptr; // hace referencial al archivo kernel.cu, aqui hay muchas funciones si es el caso
    cl_kernel kernel = nullptr;   // hace referencial a la funcion del archivo kernel.cu, se elige la que se va a utilizar
    cl_int error = 0;

    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);

    char buffer[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
    std::printf(" Using device: %s\n", buffer);

    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &error);
    commands_queue = clCreateCommandQueue(context, device_id, 0, &error);

    //--crear el programa
    const char* src_progam = kernel_code.c_str();

    program = clCreateProgramWithSource(context, 1, &src_progam, nullptr, &error);

    if(program == nullptr)
    {
        //no pudo cargar el programa
        std::printf("Can't load source program\n");
        exit(1);
    }

    error = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    if(error!=CL_SUCCESS)
    {
        size_t len;
        char buffer[1024];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

        std:printf("build error: %s\n", buffer);
    }

    kernel = clCreateKernel(program, "suma_opencl", &error);

    std::string kernel_name = "suma_opencl";
    if(kernel==nullptr)
    {
        std::printf("Can't load kernel %s\n", kernel_name.c_str());
        exit(1);
    }

    //Una vez compilado se solicita memoria tanto en el host como en el devie
    //------------------------------------------------------------------
    const size_t VECTOR_SIZE = 1024*1024*256;

    //Memoria host
    float* h_A = new float[VECTOR_SIZE];
    float* h_B = new float[VECTOR_SIZE];
    float* h_C = new float[VECTOR_SIZE];

    memset(h_C, 0, VECTOR_SIZE * sizeof(float));

    for(int i = 0 ; i < VECTOR_SIZE ; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    //--memoria device
    size_t size_in_bytes = VECTOR_SIZE * sizeof(float);
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, size_in_bytes, nullptr, &error);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, size_in_bytes, nullptr, &error);;
    cl_mem d_C =  clCreateBuffer(context, CL_MEM_READ_ONLY, size_in_bytes, nullptr, &error);;

    //--copiar host to device
    clEnqueueWriteBuffer(commands_queue, d_A, CL_TRUE, 0, size_in_bytes, h_A, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(commands_queue, d_B, CL_TRUE, 0, size_in_bytes, h_B, 0, nullptr, nullptr);

    //--Invocar el kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &VECTOR_SIZE);

    size_t glocal_work_size = VECTOR_SIZE;
    size_t local_work_size = 256;

    clEnqueueNDRangeKernel(
        commands_queue,
        kernel,
        1, //dimensiones del kernel
        0, //offset, no se desplaza nada de acuerdo al primer valor
        &glocal_work_size,
        &local_work_size,
        0,
        nullptr,
        nullptr
        );

    clFinish(commands_queue);

    //--copiar del device to host
    clEnqueueReadBuffer(commands_queue, d_C, CL_TRUE, 0, size_in_bytes, h_C, 0, nullptr, nullptr);

    for(int i = 0; i<10; i++)
    {
        std::printf(" %.0f", h_C[i]);
    }

    //liberar los recursos
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands_queue);
    clReleaseContext(context);


    return 0;
}
