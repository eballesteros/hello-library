#include <iostream>
#include <random>

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void matrixMultKernel(int * M, int * N, int * P, int I, int J, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (row<I && col<K) {
        int sumpord = 0;
        for (int j = 0; j < J; ++j) {
            sumpord += M[row*J+j] * N[j*K+col];
        }
        P[row*K+col] = sumpord;
    }
}

void matrixMult(int * M_h, int * N_h, int * P_h, int I, int J, int K) {
    // allocate memory on device and get pointers
    int *M_d, *N_d, *P_d;
    int sizeM = I*J*sizeof(int);
    int sizeN = J*K*sizeof(int);
    int sizeP = I*K*sizeof(int);
    cudaMalloc((void**)&M_d, sizeM);
    cudaMalloc((void**)&N_d, sizeN);
    cudaMalloc((void**)&P_d, sizeP);

    // copy inputs to device
    cudaMemcpy(M_d, M_h, sizeM, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, sizeN, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 dimGrid(cdiv(K, 16), cdiv(I, 16), 1);
    dim3 dimBlock(16, 16, 1);
    matrixMultKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, I, J, K);

    // copy output from device
    cudaMemcpy(P_h, P_d, sizeP, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
  // Seed the random number generator
  std::mt19937 gen(1234); // Mersenne Twister engine seeded 

  // Define the matrix shapes
  const int I = 10;
  const int J = 20;
  const int K = 15;

  // Create matrices
  int M[I*J];
  int N[J*K];
  int P[I*K];

  // init with random values
  std::uniform_int_distribution<> dist(0, 100); // Range 0 to 100
  for (int i = 0; i < I * J; ++i) {
    M[i] = dist(gen);
  }

  for (int i = 0; i < J * K; ++i) {
    N[i] = dist(gen);
  }

  for (int i = 0; i < I * K; ++i) {
   P[i] = 0;
  }

  // matrix multiply
  matrixMult(M, N, P, I, J, K);


  // Print matrix M
  std::cout << "Matrix M:\n";
  for (int i = 0; i < I; ++i) {
    for (int j = 0; j < J; ++j) {
      std::cout << M[i*J + j] << ", ";
    }
    std::cout << std::endl;
  }

  // Print matrix N
  std::cout << "\nMatrix N:\n";
  for (int i = 0; i < J; ++i) {
    for (int j = 0; j < K; ++j) {
      std::cout << N[i*K + j] << ", ";
    }
    std::cout << std::endl;
  }

  std::cout << "Matrix MxN:\n";
  for (int i = 0; i < I; ++i) {
    for (int j = 0; j < K; ++j) {
      std::cout << P[i*K + j] << ", ";
    }
    std::cout << std::endl;
  }

  return 0;
}