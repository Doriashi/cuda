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
void MatrixAdd(float* n, float* m, float* res)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int id = gridDim.x * y + x;
    res[id] = n[id] + m[id];
}


int main() {
    int r = 2;
    int s = 3;

    // Step 1
    float *h_n = new float[r * s];
    float *h_m = new float[r * s];
    float *h_res = new float[r * s];
    // Step 2
    float* d_n;
    float* d_m;
    float* d_res;
    int nbytes = r * s * sizeof(float);
    cudaMalloc(&d_n, nbytes);
    cudaMalloc(&d_m, nbytes);
    cudaMalloc(&d_res, nbytes);
    // Fill arrays
    FillMatrix(h_n, r, s, 2.0f);
	FillMatrix(h_m, r, s, 3.0f);
    

    // Step 3
    cudaMemcpy(d_n, h_n, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    ///Step 4
    dim3 num_blocks(s, r);
    //dim3 block_size(16, 16);
    MatrixAdd<<<num_blocks, 1>>>(d_n, d_m, d_res);

    cudaEventRecord(end);

    ///Step 5
    cudaMemcpy(h_res, d_res, nbytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&milliseconds, start, end);


    std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;

    // Print res matrix
    MatrixPrint(h_res, r, s);


    // Step 6
    delete[] h_n;
    delete[] h_m;
    delete[] h_res;
    
    // Step 7
    cudaFree(d_n);
    cudaFree(d_m);
    cudaFree(d_res);

    return 0;
}