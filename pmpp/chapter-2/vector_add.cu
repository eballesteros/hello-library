/*/
 * 
 * 
 *  Compile with:
 *      nvcc vector_add.cu
 * 
 *  Run with:
 *     ./a.out
 * 
/*/

#include <stdio.h>

__global__ void vecAddKernel(int* A, int* B, int* C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void vecAdd(int * A_h, int * B_h, int * C_h, int n) {
  int size = n* sizeof(int);
  int *A_d, *B_d, *C_d;

  // allocate memory on device
  cudaMalloc((void**)&A_d, size);
  cudaMalloc((void**)&B_d, size);
  cudaMalloc((void**)&C_d, size);

  // copy inputs to device
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  // launch kernel
  vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

  // copy output from device
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  // cleanup
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main() {
  // init arrays
  int N = 10000;
  int A[N], B[N], C[N];
  
  // populate arrays
  for (int i = 0; i < N; ++i) {
    A[i] = i;
    B[i] = 2*i;
  }

  // add
  vecAdd(A, B, C, N);

  // Display a few values
  // printf("C values:\n");
  // for (int i = 0; i < 5; ++i) { // Display first 5 values
  //   printf("C[%d] = %d\n", i, C[i]);
  // }
  
  return 0;
}