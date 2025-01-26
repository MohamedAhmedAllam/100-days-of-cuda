#include <cstddef>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>


void initialize_matrix(std::vector<float>& A){
    for (auto it=A.begin(); it != A.end(); ++it){
        *it = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool compare_results(const std::vector<float>& res1, const std::vector<float>& res2, float epsilon=1e-4){
    for (size_t i=0; i<res1.size(); ++i){
        //std::cout << i << " " << res1[i] << " " << res2[i] << std::endl;
        if (fabs(res1[i] - res2[i]) > epsilon){
            std::cout << i << " " << res1[i] << " " << res2[i] << std::endl;
            return false;
        }
    }
    return true;
}

void gemm_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
              size_t M, size_t N, size_t K)
{
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


__global__ void matmul_naive(float* A, float * B, float* C, size_t M, size_t N, size_t K){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < M && col < N){
        float temp = 0.0f;
        for (size_t k=0; k<K; ++k){
            temp += A[row*K+k] * B[k*N+col];
        }
        C[row*N+col] = temp;
    } 
}



void naive_cuda_gemm(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t M, size_t N, size_t K){
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32,32);
    dim3 gridSize((N+blockSize.x-1)/blockSize.x, (M+blockSize.y-1)/blockSize.y);
    matmul_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main(){
    size_t M = 2000;
    size_t N = 1000;
    size_t K = 500;

    std::vector<float> A(M*K);
    std::vector<float> B(K*N);
    std::vector<float> C(M*N);

    initialize_matrix(A);
    initialize_matrix(B);


    // Naive CUDA GEMM
    std::vector<float> C_gpu_naive(M*N);
    auto start_cuda = std::chrono::high_resolution_clock::now();
    naive_cuda_gemm(A, B, C_gpu_naive, M, N, K);
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration_cuda = end_cuda - start_cuda;
    std::cout << "Naive CUDA GEMM Time: " << duration_cuda.count() << " seconds" << std::endl;
    
    //gemm_cpu
    auto start_cpu = std::chrono::high_resolution_clock::now();
    gemm_cpu(A, B, C, M, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU GEMM Time: " << duration_cpu.count() << " seconds" << std::endl;

    if (compare_results(C, C_gpu_naive)){
        std::cout << "ALL GOOD" << std::endl;
    }  
    else{
        std::cout << "NOT EQUAL" << std::endl;
    }
    
}