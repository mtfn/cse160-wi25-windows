#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

//
// Simple OpenCL error-check helper (optional).
//
static void checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: %s (%d)\n", name, err);
        exit(EXIT_FAILURE);
    }
}

int main()
{
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(err, "clGetPlatformIDs");

    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    checkErr(err, "clGetPlatformIDs(2)");

    // Pick the first platform
    cl_platform_id platform = platforms[0];
    free(platforms);

    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    // Fallback to CPU if no GPU found
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
    }
    checkErr(err, "clGetDeviceIDs");

    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id) * numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
    }
    checkErr(err, "clGetDeviceIDs(2)");

    // Just pick the first device
    cl_device_id device = devices[0];
    free(devices);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkErr(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkErr(err, "clCreateCommandQueue");

    const char *kernelSource =    
    "__kernel void Test(__global float* X)\n"
    "{\n"
    "int x = get_global_id(0);\n"
    "int y = get_global_id(1);\n"
    "int stride = get_global_size(0);\n"
    "if (get_local_id(0) >= get_local_id(1)) {\n"
        "X[y * stride + x] = 1.0f;\n"
    "} else {\n"
        "X[y * stride + x] = 0.0f;\n"
    "}\n"
    "}\n";


    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkErr(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build log if there's an error
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *log = (char *)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel = clCreateKernel(program, "Test", &err);
    checkErr(err, "clCreateKernel");

    int image_width = 16;
    int image_height = 16;
    size_t bytes = image_width * image_height * sizeof(float);

    float *hostData = (float *)malloc(bytes);

    cl_mem deviceBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    checkErr(err, "clCreateBuffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceBuffer);
    checkErr(err, "clSetKernelArg(0)");

    size_t globalSize[2] = { (size_t)image_width, (size_t)image_height };
    size_t localSize[2]  = { 4, 4 };

    err = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 2,         // work_dim (2D)
                                 NULL,      // global_work_offset
                                 globalSize,// global_work_size
                                 localSize, // local_work_size (optional)
                                 0,
                                 NULL,
                                 NULL);
    checkErr(err, "clEnqueueNDRangeKernel");

    clFinish(queue);

    err = clEnqueueReadBuffer(queue, deviceBuffer, CL_TRUE, 0, bytes, hostData, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");

    printf("Result array:\n");
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            printf("%.0f ", hostData[j * image_width + i]);
        }
        printf("\n");
    }

    free(hostData);
    clReleaseMemObject(deviceBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

