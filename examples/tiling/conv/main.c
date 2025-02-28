#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// Error-check helper function
static void checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: %s (%d)\n", name, err);
        exit(EXIT_FAILURE);
    }
}

// Function to read kernel file
char* loadKernelSource(const char *filename, size_t *kernelSize)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    // Seek to end to determine file size
    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    rewind(fp);
    
    char *source = (char*)malloc(length + 1);
    if (!source) {
        fprintf(stderr, "Failed to allocate memory for kernel source.\n");
        exit(EXIT_FAILURE);
    }
    
    size_t readLength = fread(source, 1, length, fp);
    source[readLength] = '\0'; // Null-terminate the string
    if (kernelSize)
        *kernelSize = readLength;
    
    fclose(fp);
    return source;
}



int main(void)
{

    int H = 32;
    int W = 32;
    int K = 3;
    int OUT_H = H-K+1;
    int OUT_W = W-K+1;


    cl_int err;
    
    // 1. Get the first available platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    checkErr(err, "clGetPlatformIDs");
    
    // 2. Get the first available device (CPU/GPU)
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    checkErr(err, "clGetDeviceIDs");
    
    // 3. Create a context and command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkErr(err, "clCreateContext");
    
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkErr(err, "clCreateCommandQueue");
    
    // 4. Create and build the OpenCL program
    
    size_t kernelSize;
    char *kernelSource = loadKernelSource("kernel.cl", &kernelSize);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkErr(err, "clCreateProgramWithSource");
    
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build log if there's an error
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*) malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
    
    // 5. Create the kernel
    cl_kernel kernel_cl = clCreateKernel(program, "conv2d", &err);
    checkErr(err, "clCreateKernel");
    
    // 6. Allocate and initialize host memory
    float *h_image   = (float*) malloc(sizeof(float) * H * W);
    float *h_kernel  = (float*) malloc(sizeof(float) * K * K);
    float *h_output  = (float*) malloc(sizeof(float) * OUT_H * OUT_W);
    
    for (int i = 0; i < H * W; i++) {
        h_image[i] = 1.0f; // fill image with 1's
    }
    for (int i = 0; i < K * K; i++) {
        h_kernel[i] = 3.0f; // fill kernel with 2's
    }
    
    // 7. Create device buffers
    cl_mem d_image = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * H * W, h_image, &err);
    checkErr(err, "clCreateBuffer(d_image)");
    
    cl_mem d_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * K * K, h_kernel, &err);
    checkErr(err, "clCreateBuffer(d_kernel)");
    
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     sizeof(float) * OUT_H * OUT_W, NULL, &err);
    checkErr(err, "clCreateBuffer(d_output)");
    
    // 8. Set the kernel arguments
    err = clSetKernelArg(kernel_cl, 0, sizeof(cl_mem), &d_image);
    checkErr(err, "clSetKernelArg(0)");
    err = clSetKernelArg(kernel_cl, 1, sizeof(cl_mem), &d_kernel);
    checkErr(err, "clSetKernelArg(1)");
    err = clSetKernelArg(kernel_cl, 2, sizeof(cl_mem), &d_output);
    checkErr(err, "clSetKernelArg(2)");
    err = clSetKernelArg(kernel_cl, 3, sizeof(int), &H);
    checkErr(err, "clSetKernelArg(3)");
    err = clSetKernelArg(kernel_cl, 4, sizeof(int), &W);
    checkErr(err, "clSetKernelArg(4)");
    err = clSetKernelArg(kernel_cl, 5, sizeof(int), &K);
    checkErr(err, "clSetKernelArg(5)");
    
    // 9. Define the NDRange (output image dimensions)
    size_t global[2] = {OUT_W, OUT_H};
    
    // 10. Enqueue the kernel for execution
    err = clEnqueueNDRangeKernel(queue, kernel_cl, 2, NULL, global, NULL, 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel");
    
    // 11. Read the result back to host memory
    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                              sizeof(float) * OUT_H * OUT_W, h_output, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");
    
    // 12. Print the output image
    printf("Output image (size %dx%d):\n", OUT_H, OUT_W);
    for (int i = 0; i < OUT_H; i++) {
        for (int j = 0; j < OUT_W; j++) {
            printf("%6.2f ", h_output[i * OUT_W + j]);
        }
        printf("\n");
    }
    
    // 13. Clean up resources
    clReleaseMemObject(d_image);
    clReleaseMemObject(d_kernel);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel_cl);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(h_image);
    free(h_kernel);
    free(h_output);
    
    return 0;
}

