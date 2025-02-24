#ifndef SRC_LAYER_OPENCL_H
#define SRC_LAYER_OPENCL_H

#define KERNEL_PATH "src/layer/custom/new-forward-kernel.cl"

#include "kernel.h"
#include "device.h"

class OpenCL
{
    public:
        cl_program program;        // program
        cl_kernel kernel;          // kernel
        cl_command_queue queue;    // command queue
        cl_context context;        // context

        void setup(cl_device_type device_type);
        void teardown();
};

#endif