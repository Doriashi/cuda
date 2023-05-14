#include <iostream>

void FillVector(float* arr, int s, float num) {
    for (int j = 0; j < s; ++j) {
        arr[j] = num;
    }
}

void VectorPrint(float *arr, int r) {
	for (int i = 0; i < r; ++i) {
        std::cout << arr[i] << "\n";
	}
}

__global__
void VectorMul(int n, float* x, float* y, float* res) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < n; index += stride) {
        res[index] = x[index] * y[index];
    }
}


int main() {
    int n = 5;
    // Step 1
    float* h_x = new float[n];
    float* h_y = new float[n];
    float* h_res = new float[n];
    // Step 2
    float* d_x;
    float* d_y;
    float* d_res;
    int nbytes = n * sizeof(float);
    cudaMalloc(&d_x, nbytes);
    cudaMalloc(&d_y, nbytes);
    cudaMalloc(&d_res, nbytes);
    // Fill arrays
    FillVector(h_x, n, 2.0f);
    FillVector(h_y, n, 3.0f);

    // Step 3
    cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    // Step 4
    VectorMul<<<1, 16>>>(n, d_x, d_y, d_res);

    cudaEventRecord(end);

    // Step 5
    cudaMemcpy(h_res, d_res, nbytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&milliseconds, start, end);


    std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;
    
    // Print res vector
    VectorPrint(h_res, n);


    // Step 6
    delete[] h_x;
    delete[] h_y;
    delete[] h_res;
    // Step 7
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);
}