#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

// globals
int CHANNELS = 3;

// 2D blur kernel
__global__ void image_blur_kernel_2d(unsigned char* x, unsigned char* out, int width, int height) {
    int BLUR_SIZE = 7; // TODO make global
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int grayIdx = row*width + col;
        int rgbIdx = 3*grayIdx;
        int rVal = 0;
        int gVal = 0;
        int bVal = 0;
        int pixels = 0;
        for (int colDiff=-BLUR_SIZE; colDiff<BLUR_SIZE+1; ++colDiff) {
            for (int rowDiff=-BLUR_SIZE; rowDiff<BLUR_SIZE+1; ++rowDiff) {
                int curCol = col + colDiff;
                int curRow = row + rowDiff;
                if (curCol>=0 && curCol<width && curRow>=0 && curRow<height) {
                    int curGrayIdx = curRow*width + curCol;
                    int curRgbIdx = 3*curGrayIdx;
                    rVal += x[curRgbIdx  ];
                    gVal += x[curRgbIdx+1];
                    bVal += x[curRgbIdx+2];
                    ++pixels;
                }
            }
        }
        out[rgbIdx  ] = (unsigned char) (rVal/pixels);
        out[rgbIdx+1] = (unsigned char) (gVal/pixels);
        out[rgbIdx+2] = (unsigned char) (bVal/pixels);
    }
}

// 3D blur kernel (meaning 2 dimensional blur, implemented as a 3D kernel)
__global__ void image_blur_kernel_3d(unsigned char* x, unsigned char* out, int width, int height, int channels) {
    int BLUR_SIZE = 7; // TODO make global
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int plane = blockIdx.z*blockDim.z + threadIdx.z;

    if (col<width && row<height && plane<channels) {
        int rgbIdx = 3*(row*width + col) + plane;
        int pixVal = 0;
        int pixels = 0;
        for (int colDiff=-BLUR_SIZE; colDiff<BLUR_SIZE+1; ++colDiff) {
            for (int rowDiff=-BLUR_SIZE; rowDiff<BLUR_SIZE+1; ++rowDiff) {
                int curCol = col + colDiff;
                int curRow = row + rowDiff;
                if (curCol>=0 && curCol<width && curRow>=0 && curRow<height) {
                    pixVal += x[3*(curRow*width + curCol) + plane];
                    ++pixels;
                }
            }
        }
        out[rgbIdx] = (unsigned char) (pixVal/pixels);
    }
}


torch::Tensor image_blur(torch::Tensor input_d) {
    CHECK_INPUT(input_d);
    
    int height = input_d.size(0);
    int width = input_d.size(1);
    int channels = input_d.size(2);

    if (channels != 3) {
        throw std::runtime_error("Error: Expected 3 channels");
    }

    auto output_d = torch::empty({height,width,channels}, input_d.options());

    
    // A. 2D
    // dim3 dimGrid(cdiv(width, 16), cdiv(height, 16), 1);
    // dim3 dimBlock(16, 16, 1);
    // image_blur_kernel_2d<<<dimGrid, dimBlock>>>(
    //     input_d.data_ptr<unsigned char>(), output_d.data_ptr<unsigned char>(), width, height
    // );
    
    // B. 3D
    dim3 dimGrid(cdiv(width, 8), cdiv(height, 8), cdiv(channels, 4));
    dim3 dimBlock(8, 8, 4);
    image_blur_kernel_3d<<<dimGrid, dimBlock>>>(
        input_d.data_ptr<unsigned char>(), output_d.data_ptr<unsigned char>(), width, height, channels
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output_d;
}