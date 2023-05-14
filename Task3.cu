#include <iostream>

void FillMatrix(float* arr, int r, int s, float num) {
	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < s; ++j) {
            arr[i * s + j] = num;
		}
	}
}

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
void MatrixMulVector(float* m, float* v, float* res, int s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    res[j] = .0f;

    for (int k = 0; k < s; ++k) {
        res[j] += m[i * s + k] * v[j];
    }
}


int main() {
    int r = 3;
    int s = 6;

    // Step 1
    float *h_m = new float[r * s];
    float *h_v = new float[s];
    float *h_res = new float[r];
    // Step 2
    float* d_m;
    float* d_v;
    float* d_res;
    cudaMalloc(&d_m, r * s * sizeof(float));
    cudaMalloc(&d_v, s * sizeof(float));
    cudaMalloc(&d_res, r * sizeof(float));
    // Fill arrays
    FillMatrix(h_m, r, s, 2.0f);
	FillVector(h_v, s, 3.0f);
    

    // Step 3
    cudaMemcpy(d_m, h_m, r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, s * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    ///Step 4
    dim3 num_blocks(r, s);
    MatrixMulVector<<<num_blocks, 1>>>(d_m, d_v, d_res, s);

    cudaEventRecord(end);

    ///Step 5
    cudaMemcpy(h_res, d_res, r * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&milliseconds, start, end);


    std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;

    // Print res vector
    VectorPrint(h_res, r);


    // Step 6
    delete[] h_m;
    delete[] h_v;
    delete[] h_res;
    
    // Step 7
    cudaFree(d_m);
    cudaFree(d_v);
    cudaFree(d_res);

    return 0;
}