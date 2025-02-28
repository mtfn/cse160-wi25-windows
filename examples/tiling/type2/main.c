#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define TILE_WIDTH 16

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

    int H=18, W=18, K=3;
    int H_OUT = H-K+1;
    int W_OUT = W-K+1;  


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
    
    // 4. Create and build the program
    size_t kernelSize;
    char *kernelSource = loadKernelSource("kernel.cl", &kernelSize);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkErr(err, "clCreateProgramWithSource");

    char buildOptions [300];
    sprintf(buildOptions, "-DTILE_WIDTH=%d -DR=%d", TILE_WIDTH, K/2);
    
    err = clBuildProgram(program, 1, &device, buildOptions, NULL, NULL);
    if (err != CL_SUCCESS) {
        // If there is a build error, print the build log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*) malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
    
    // 5. Create the kernel
    cl_kernel kernel = clCreateKernel(program, "conv_forward_kernel", &err);
    checkErr(err, "clCreateKernel");
    
    // 6. Allocate and initialize host memory
    float *h_x = (float*) malloc(sizeof(float) * H * W);
    float *h_mask = (float*) malloc(sizeof(float) * K * K);
    float *h_y = (float*) malloc(sizeof(float) * H_OUT * W_OUT);
    
    // Fill input image with 1's
    for (int i = 0; i < H * W; i++) {
        h_x[i] = 1.0f;
    }
    // Fill convolution kernel (mask) with 3's
    for (int i = 0; i < K * K; i++) {
        h_mask[i] = 2.0f;
    }
    
    // 7. Create device buffers
    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * H * W, h_x, &err);
    checkErr(err, "clCreateBuffer(d_x)");
    
    cl_mem d_mask = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * K * K, h_mask, &err);
    checkErr(err, "clCreateBuffer(d_mask)");
    
    cl_mem d_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                sizeof(float) * H_OUT * W_OUT, NULL, &err);
    checkErr(err, "clCreateBuffer(d_y)");
    
    // 8. Set kernel arguments
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_y);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_x);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_mask);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &H);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &W);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &K);
    checkErr(err, "clSetKernelArg");
    
    // 9. Determine work-group sizes.
    // For correct tiling, the local work-group size should be:
    //    (TILE_WIDTH + 2*R, TILE_WIDTH + 2*R) where R = K/2.
    const int R = K/2; // equals 1
    size_t localWorkSize[3] = {TILE_WIDTH + 2 * R, TILE_WIDTH + 2 * R, 1}; // {18, 18, 1}
    // We have one output tile covering the full output image (16x16)
    size_t numGroups0 = 1; // group 0 (unused index m)
    size_t numGroups1 = 1; // group 1 (tile index)
    size_t numGroups2 = 1; // group 2 (batch index)
    size_t globalWorkSize[3] = {
        localWorkSize[0] * numGroups0,
        localWorkSize[1] * numGroups1,
        localWorkSize[2] * numGroups2
    }; // {18, 18, 1}
    
    // 10. Enqueue the kernel for execution
    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalWorkSize, localWorkSize,
                                 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel");
    
    // 11. Read the result back to host memory
    err = clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, sizeof(float) * H_OUT * W_OUT,
                              h_y, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");
    
    // 12. Print the output image.
    // For our settings, each output pixel should equal the sum over a 3x3 region:
    //    9 * (1 * 3) = 27.
    printf("Output image (%d x %d):\n", H_OUT, W_OUT);
    for (int i = 0; i < H_OUT; i++) {
        for (int j = 0; j < W_OUT; j++) {
            printf("%6.2f ", h_y[i * W_OUT + j]);
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

