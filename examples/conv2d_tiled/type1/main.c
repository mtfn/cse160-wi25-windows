#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "../../opencl_utils.h"

#define TILE_WIDTH 16

int main(void)
{
    // Image dimensions and kernel size.
    int H = 18, W = 18, K = 3; // With K = 3, we have R = 1.
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    
    cl_int err;
    OpenCL ocl;

    char buildOptions [300]; 
    sprintf(buildOptions, "-DTILE_WIDTH=%d -DR=%d", TILE_WIDTH, K/2);
    opencl_setup(&ocl, "conv_forward_kernel", "kernel.cl", buildOptions);

    // Allocate and initialize host memory.
    float *h_x = (float*) malloc(sizeof(float) * H * W);
    float *h_mask = (float*) malloc(sizeof(float) * K * K);
    float *h_y = (float*) malloc(sizeof(float) * H_out * W_out);

    // Fill the input image with 1's.
    for (int i = 0; i < H * W; i++) {
        h_x[i] = 1.0f;
    }
    // Fill the convolution mask with 2's.
    for (int i = 0; i < K * K; i++) {
        h_mask[i] = 2.0f;
    }

    // Create device buffers.
    cl_mem d_x = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * H * W, h_x, &err);
    checkErr(err, "clCreateBuffer(d_x)");
    cl_mem d_mask = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * K * K, h_mask, &err);
    checkErr(err, "clCreateBuffer(d_mask)");
    cl_mem d_y = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY,
                                  sizeof(float) * H_out * W_out, NULL, &err);
    checkErr(err, "clCreateBuffer(d_y)");

    // Set kernel arguments.
    err  = clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), &d_y);
    err |= clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), &d_x);
    err |= clSetKernelArg(ocl.kernel, 2, sizeof(cl_mem), &d_mask);
    err |= clSetKernelArg(ocl.kernel, 3, sizeof(int), &H);
    err |= clSetKernelArg(ocl.kernel, 4, sizeof(int), &W);
    err |= clSetKernelArg(ocl.kernel, 5, sizeof(int), &K);
    checkErr(err, "clSetKernelArg");

    // Define NDRange.
    // Each work-group computes one output tile of size TILE_WIDTH x TILE_WIDTH.
    unsigned int gridX = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    unsigned int gridY = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    size_t localWorkSize[2] = {TILE_WIDTH, TILE_WIDTH};
    size_t globalWorkSize[2] = {gridX * TILE_WIDTH, gridY * TILE_WIDTH};

    // Enqueue the kernel.
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 2, NULL, globalWorkSize, localWorkSize,
                                 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel");

    // Read the result back to host memory.
    err = clEnqueueReadBuffer(ocl.queue, d_y, CL_TRUE, 0, sizeof(float) * H_out * W_out,
                              h_y, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");

    // Print the output image.
    // For our settings, each output pixel equals the sum over a 3x3 region:
    //   9 * (1*2) = 18.
    printf("Output image (%d x %d):\n", H_out, W_out);
    for (int i = 0; i < H_out; i++) {
        for (int j = 0; j < W_out; j++) {
            printf("%6.2f ", h_y[i * W_out + j]);
        }
        printf("\n");
    }

    // Clean up.
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_mask);
    clReleaseMemObject(d_y);
    opencl_cleanup(&ocl);

    free(h_x);
    free(h_mask);
    free(h_y);

    return 0;
}
