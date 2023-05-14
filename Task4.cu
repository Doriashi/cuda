#include <iostream>

void FillMatrix(float* arr, int r, int s, float num) {
	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < s; ++j) {
            arr[i * s + j] = num;
		}
	}
}

void MatrixPrint(float *arr, int r, int s) {
	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < s; ++j) {
			std::cout << arr[i * s + j] << " ";
		}
        std::cout << "\n";
	}
}

__global__
void MatrixMulMatrix(float* m, float* n, float* res, int mid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int width = blockDim.y * gridDim.y;

    res[i * width + j] = .0f;

    for (int k = 0; k < mid; ++k) {
        res[i * width + j] += m[i * mid + k] * n[k * width + j];
    }
}


int main() {
    int r = 3;
    int s = 4;
    int mid = 5;

    // Step 1
    float *h_m = new float[r * mid];
    float *h_n = new float[s * mid];
    float *h_res = new float[r * s];
    // Step 2
    float* d_m;
    float* d_n;
    float* d_res;
    cudaMalloc(&d_m, r * mid * sizeof(float));
    cudaMalloc(&d_n, s * mid * sizeof(float));
    cudaMalloc(&d_res, r * s * sizeof(float));
    // Fill arrays
    FillMatrix(h_m, r, mid, 1.0f);
	FillMatrix(h_n, mid, s, 1.0f);
    

    // Step 3
    cudaMemcpy(d_m, h_m, r * mid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, s * mid * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    ///Step 4
    dim3 num_blocks(r, s);
    dim3 block_size(r, s);
    MatrixMulMatrix<<<num_blocks, 1>>>(d_m, d_n, d_res, mid);

    cudaEventRecord(end);

    ///Step 5
    cudaMemcpy(h_res, d_res, r * s * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&milliseconds, start, end);


    std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;

    // Print res vector
    MatrixPrint(h_res, r, s);


    // Step 6
    delete[] h_m;
    delete[] h_n;
    delete[] h_res;
    
    // Step 7
    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_res);

    return 0;
}