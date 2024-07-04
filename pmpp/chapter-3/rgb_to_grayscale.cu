#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}


// Channels first kernels
__global__ void rgb_to_grayscale_kernel_2d_cf(unsigned char* x, unsigned char* out, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int n = width*height;
    if (col < width && row < height) {
        int i = row*width + col;
        int r = x[i];
        int g = x[i + n];
        int b = x[i + 2*n];
        out[i] = 0.21*r + 0.71*g + 0.07*b;
    }
}

__global__ void rgb_to_grayscale_kernel_1d_cf(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < n) {
        int r = x[i];
        int g = x[i + n];
        int b = x[i + 2*n];
        out[i] = 0.21*r + 0.71*g + 0.07*b;
    }
}

// Channels last kernels
__global__ void rgb_to_grayscale_kernel_2d_cl(unsigned char* x, unsigned char* out, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int n = width*height;
    if (col < width && row < height) {
        int grayIdx = row*width + col;
        int rgbIdx = 3*grayIdx;
        int r = x[rgbIdx];
        int g = x[rgbIdx + 1];
        int b = x[rgbIdx + 2];
        out[grayIdx] = 0.21*r + 0.71*g + 0.07*b;
    }
}

__global__ void rgb_to_grayscale_kernel_1d_cl(unsigned char* x, unsigned char* out, int n) {
    int grayIdx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (grayIdx < n) {
        int rgbIdx = 3 * grayIdx;
        int r = x[rgbIdx];
        int g = x[rgbIdx + 1];
        int b = x[rgbIdx + 2];
        out[grayIdx] = 0.21*r + 0.71*g + 0.07*b;
    }
}

torch::Tensor rgb_to_grayscale(torch::Tensor input_d) {
    CHECK_INPUT(input_d);
    // -- A. Channels last (CL)
    int height = input_d.size(0);
    int width = input_d.size(1);
    int channels = input_d.size(2);

    if (channels != 3) {
        throw std::runtime_error("Error: Expected 3 channels");
    }

    auto output_d = torch::empty({height,width}, input_d.options());

    // -- A1. 1D CL kernel
    // int n = width*height;
    // dim3 dimGrid(cdiv(n, 256), 1, 1);
    // dim3 dimBlock(256, 1, 1);
    // rgb_to_grayscale_kernel_1d_cl<<<dimGrid, dimBlock>>>(
    //     input_d.data_ptr<unsigned char>(), output_d.data_ptr<unsigned char>(), n
    // );
    
    // -- A2. 2D CL kernel
    dim3 dimGrid(cdiv(width, 16), cdiv(height, 16), 1);
    dim3 dimBlock(16, 16, 1);
    rgb_to_grayscale_kernel_2d_cl<<<dimGrid, dimBlock>>>(
        input_d.data_ptr<unsigned char>(), output_d.data_ptr<unsigned char>(), width, height
    );

    // -- B. Channels first (CF)
    // int width = input_d.size(1);
    // int height = input_d.size(2);

    // auto output_d = torch::empty({width,height}, input_d.options());

    // -- B1. 1D CF kernel
    // int n = width*height;
    // dim3 dimGrid(cdiv(n, 256), 1, 1);
    // dim3 dimBlock(256, 1, 1);
    // rgb_to_grayscale_kernel_1d_cf<<<dimGrid, dimBlock>>>(
    //     input_d.data_ptr<unsigned char>(), output_d.data_ptr<unsigned char>(), n
    // );
    
    // -- B2. 2D CF kernel
    // dim3 dimGrid(cdiv(width, 16), cdiv(height, 16), 1);
    // dim3 dimBlock(16, 16, 1);
    // rgb_to_grayscale_kernel_2d_cf<<<dimGrid, dimBlock>>>(
    //     input_d.data_ptr<unsigned char>(), output_d.data_ptr<unsigned char>(), width, height
    // );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output_d;
}