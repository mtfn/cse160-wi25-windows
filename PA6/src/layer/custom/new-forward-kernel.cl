#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void do_not_remove_this_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void prefn_marker_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}



__kernel void conv_forward_kernel(__global float *y, __constant float *x, __constant float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
	//@@ Insert code to implement convolution here
}
