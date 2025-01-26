#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// Function to initialize matrices with random values
void initialize_matrix(std::vector<float>& matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// CPU implementation of GEMM
void gemm_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
              size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CUDA error checking macro
#define cudaCheckError() {                                                  \
    cudaError_t e=cudaGetLastError();                                         \
    if(e!=cudaSuccess) {                                                     \
        std::cerr << "CUDA Error " << cudaGetErrorString(e) << " at line " << __LINE__ << std::endl;  \
        exit(EXIT_FAILURE);                                                   \
    }                                                                        \
}

// Placeholder for the CUDA GEMM kernel
__global__ void gemm_cuda_kernel(float* A, float* B, float* C, size_t M, size_t N, size_t K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    //printf("%d\n", idx);

    if (row < M && col < N){
        float temp = 0.0f;
        for (int k=0; k<K;k++){
            //printf("%d\n", k);
            temp += A[row*K + k] * B[k*N + col];
        }
        C[row*N + col] = temp;
    }
}

// Wrapper for the CUDA GEMM kernel
void gemm_cuda(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
               size_t M, size_t N, size_t K) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    gemm_cuda_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaCheckError();

    // Copy result from device to host
    cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

// Function to compare two matrices for accuracy
bool compare_matrices(const std::vector<float>& C1, const std::vector<float>& C2, float epsilon = 1e-5) {
    for (size_t i = 0; i < C1.size(); ++i) {
        if (fabs(C1[i] - C2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main() {
    size_t M = 512;  // Number of rows in A and C
    size_t N = 512;  // Number of columns in B and C
    size_t K = 512;  // Number of columns in A and rows in B

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_cpu(M * N);
    std::vector<float> C_cuda(M * N);

    // Initialize matrices
    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);

    // CPU GEMM
    auto start_cpu = std::chrono::high_resolution_clock::now();
    gemm_cpu(A, B, C_cpu, M, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration_cpu = end_cpu - start_cpu;

    std::cout << "CPU GEMM Time: " << duration_cpu.count() << " seconds" << std::endl;

    // CUDA GEMM
    auto start_cuda = std::chrono::high_resolution_clock::now();
    gemm_cuda(A, B, C_cuda, M, N, K);
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration_cuda = end_cuda - start_cuda;

    std::cout << "CUDA GEMM Time: " << duration_cuda.count() << " seconds" << std::endl;

    // Compare results
    if (compare_matrices(C_cpu, C_cuda)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
