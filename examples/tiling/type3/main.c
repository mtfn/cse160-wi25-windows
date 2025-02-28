#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define TILE_WIDTH 16

// Error-check helper function
static void checkErr(cl_int err, const char *name) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: %s (%d)\n", name, err);
        exit(EXIT_FAILURE);
    }
}

// Function to read kernel file
char* loadKernelSource(const char *filename, size_t *kernelSize) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    rewind(fp);
    char *source = (char*)malloc(length + 1);
    if (!source) {
        fprintf(stderr, "Failed to allocate memory for kernel source.\n");
        exit(EXIT_FAILURE);
    }
    size_t readLength = fread(source, 1, length, fp);
    source[readLength] = '\0';
    if (kernelSize)
        *kernelSize = readLength;
    fclose(fp);
    return source;
}

int main(void) {
    int H = 18, W = 18, K = 3;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    
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
    
    // 4. Create and build the program from "kernel.cl"
    size_t kernelSize;
    char *kernelSource = loadKernelSource("kernel.cl", &kernelSize);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkErr(err, "clCreateProgramWithSource");
    
    // Only need to define TILE_WIDTH for this strategy.
    char buildOptions[300];
    sprintf(buildOptions, "-DTILE_WIDTH=%d", TILE_WIDTH);
    
    err = clBuildProgram(program, 1, &device, buildOptions, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*) malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
    
    // 5. Create the kernel from the program.
    cl_kernel kernel = clCreateKernel(program, "conv_forward_kernel", &err);
    checkErr(err, "clCreateKernel");
    
    // 6. Allocate and initialize host memory
    float *h_x = (float*) malloc(sizeof(float) * H * W);
    float *h_mask = (float*) malloc(sizeof(float) * K * K);
    float *h_y = (float*) malloc(sizeof(float) * H_out * W_out);
    
    // Fill input image with 1's
    for (int i = 0; i < H * W; i++) {
        h_x[i] = 1.0f;
    }
    // Fill convolution kernel (mask) with 2's (so each output should be 9*2 = 18)
    for (int i = 0; i < K * K; i++) {
        h_mask[i] = 3.0f;
    }
    
    // 7. Create device buffers
    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * H * W, h_x, &err);
    checkErr(err, "clCreateBuffer(d_x)");
    
    cl_mem d_mask = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * K * K, h_mask, &err);
    checkErr(err, "clCreateBuffer(d_mask)");
    
    cl_mem d_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * H_out * W_out, NULL, &err);
    checkErr(err, "clCreateBuffer(d_y)");
    
    // 8. Set kernel arguments.
    // Note: We pass two extra arguments: the number of tiles in X and Y (W_grid and H_grid).
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_y);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_x);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_mask);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &H);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &W);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &K);
    
    unsigned int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    unsigned int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &W_grid);
    err |= clSetKernelArg(kernel, 7, sizeof(unsigned int), &H_grid);
    checkErr(err, "clSetKernelArg");
    
    // 9. Determine work-group sizes (using 2D NDRange)
    size_t localWorkSize[2] = {TILE_WIDTH, TILE_WIDTH};
    size_t globalWorkSize[2] = {TILE_WIDTH * W_grid, TILE_WIDTH * H_grid};
    
    // 10. Enqueue the kernel for execution
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize,
                                 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel");
    
    // 11. Read the result back to host memory
    err = clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, sizeof(float) * H_out * W_out,
                              h_y, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");
    
    // 12. Print the output image.
    // With the given settings each output pixel should be: 3x3 region = 9 * (1*2) = 18.
    printf("Output image (%d x %d):\n", H_out, W_out);
    for (int i = 0; i < H_out; i++) {
        for (int j = 0; j < W_out; j++) {
            printf("%6.2f ", h_y[i * W_out + j]);
        }
        printf("\n");
    }
    
    // 13. Clean up resources
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_mask);
    clReleaseMemObject(d_y);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(h_x);
    free(h_mask);
    free(h_y);
    
    return 0;
}

