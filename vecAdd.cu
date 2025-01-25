#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>


void initialize_vector(std::vector<float>& x){
    for (auto it = x.begin(); it != x.end(); ++it){
        *it = static_cast<float>(rand()) / RAND_MAX;
    }
}

void vecAdd_CPU(const std::vector<float>& x, const std::vector<float>& y, std::vector<float>& z){
    for (int r=0; r<x.size(); ++r){
        z[r] = x[r] + y[r];
    }
}


bool compare_results(const std::vector<float>& res1, const std::vector<float>& res2, float epsilon = 1e-5){
    for (int r=0; r<res1.size(); ++r){
        if (fabs(res1[r] - res2[r]) > epsilon){
            std::cout << r << " " << res1[r] << " " << res2[r] << std::endl;
            return false;
        }
    }
    return true;
}


__global__ void vecAdd(float* x, float* y, float* z, size_t s){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < s){
        z[idx] = x[idx] + y[idx];
    }
}

int main(){

    int size = 100000000;

    std::vector<float> x(size);
    std::vector<float> y(size);
    std::vector<float> z_cpu(size);
    std::vector<float> z_gpu(size);

    initialize_vector(x);
    initialize_vector(y);

    //for (auto it = x.begin(); it != x.end(); ++it){
    //    std::cout << *it << std::endl;
    //}

    float* x_d;
    float* y_d;
    float* z_d;

    cudaMalloc(&x_d, sizeof(float) * size);
    cudaMalloc(&y_d, sizeof(float) * size);
    cudaMalloc(&z_d, sizeof(float) * size);

    cudaMemcpy(x_d, x.data(), sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y.data(), sizeof(float) * size, cudaMemcpyHostToDevice);

    int blockDim=1024;

    vecAdd_CPU(x, y, z_cpu);
    vecAdd<<<(size+blockDim-1) / blockDim, blockDim>>>(x_d, y_d, z_d, x.size());

    cudaMemcpy(z_gpu.data(), z_d, sizeof(float) * size, cudaMemcpyDeviceToHost);


    if (compare_results(z_gpu, z_cpu)){
        std::cout << "All GOOD" << std::endl;
    }
    else{
        std::cout << "NOT EQUAL" << std::endl;
    }

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);

}