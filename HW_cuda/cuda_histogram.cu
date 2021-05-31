
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>

using namespace std;

// CUDA KERNEL FUNCTIONS

__global__ void Hello()
{
    //int globalidx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int globalidx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("blockIdx.x %d\n",blockIdx.x);
    printf("Hello from blockx = %d\t tx = %d\t ty=%d\t tz=%d\t gi=%d\n",blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z, globalidx);  
}

__global__ void Histogram(int hist_array_1D_size,int array_1D_size, int *hist_array_1D_cuda, int *array_1D_cuda)
{
    int globalidx = blockIdx.x * blockDim.x + threadIdx.x; 
    int N = hist_array_1D_size;
    // printf("Hello from blockx = %d\t tx = %d\t ty=%d\t tz=%d\t gi=%d\n",blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z, globalidx);  
    
    if(globalidx<N)
    {   
        int temp = 0;
        for (int i = 0; i < array_1D_size; i++)
        {
            if (array_1D_cuda[i] == globalidx)
            {
                temp++;
            }
        }
        
        hist_array_1D_cuda[globalidx] = temp;
        
    } 
}


// CPU FUNCTIONS

int ** malloc_matrix(int N)
{
    int ** matrix = (int **)malloc(N * sizeof(int *));
    
    for (int i = 0; i < N; ++i)
    {   
        matrix[i] = (int *)malloc(N * sizeof(int));
    }
    
    return matrix;
}

float ** malloc_matrix_float(int N)
{
    float ** matrix = (float **)malloc(N * sizeof(float *));
    
    for (int i = 0; i < N; ++i)
    {   
        matrix[i] = (float *)malloc(N * sizeof(float));
    }
    
    return matrix;
}

void print_matrix(int * matrix, int N)
{
    for (int j = 0; j < N; j++)
    {
        cout << matrix[j] << " ";
    }
    cout << endl;
}

void print_matrix(int ** matrix, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void print_matrix(float ** matrix, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void print_matrix(float * matrix, int N)
{
    for (int j = 0; j < N; j++)
    {
        cout << matrix[j] << " ";
    }
    cout << endl;
}

void convert_1d_to_2d (int array_size, int * array_1D)
{
    int ** array_2D;
    int array_2D_size = sqrt(array_size);
    array_2D = malloc_matrix(array_2D_size);

    for (int i = 0; i < array_2D_size; i++)
    {
        for (int j = 0; j < array_2D_size; j++)
        {
            array_2D[i][j] = array_1D[i*array_2D_size + j];
        }
    }
    
    print_matrix(array_2D,array_2D_size);
}

void convert_1d_to_2d_float (int array_size, float * array_1D)
{
    float ** array_2D;
    int array_2D_size = sqrt(array_size);
    array_2D = malloc_matrix_float(array_2D_size);

    for (int i = 0; i < array_2D_size; i++)
    {
        for (int j = 0; j < array_2D_size; j++)
        {
            array_2D[i][j] = array_1D[i*array_2D_size + j];
        }
    }
  
    print_matrix(array_2D,array_2D_size);


}


// MAIN FUNCTION
int main(int argc, char *argv[]){
    
    // IMAGE TO ARRAY
    ifstream file("image_array.txt");
    vector<int> image;
    if(file.is_open())
    {   
        while (!file.eof()) 
        {
            int temp;
            file >> temp;
            image.push_back(temp);
        } 
    }
    file.close();
    image.pop_back();
    
    // ARRAY CREATION
    int array_1D_size = image.size();
    int *array_1D = (int*)malloc(sizeof(int)*array_1D_size);

    for (int i = 0; i < array_1D_size; i++)
    {
        array_1D[i] = image[i];
    }

    // convert_1d_to_2d(array_1D_size, array_1D);
 
    // HISTOGRAMM ARRAY CREATION
    int hist_array_1D_size = 226;
    int *hist_array_1D = (int*)malloc(sizeof(int)*hist_array_1D_size);

    for (int i = 0; i < hist_array_1D_size; i++)
    {
        hist_array_1D[i] = 0;
    }

    
    // ---------- CUDA ZONE -------------

    // printf("This is done by CPU before CUDA threads\n");

    // ARRAY TO DEVICE
    int *array_1D_cuda;
    cudaMalloc(&array_1D_cuda, sizeof(int)*array_1D_size);
    cudaMemcpy(array_1D_cuda, array_1D, array_1D_size * sizeof(int), cudaMemcpyHostToDevice);

    // HSTOGRAMM ARRAY TO DEVICE
    int *hist_array_1D_cuda;
    cudaMalloc(&hist_array_1D_cuda, sizeof(int)*hist_array_1D_size);
    cudaMemcpy(hist_array_1D_cuda, hist_array_1D, hist_array_1D_size * sizeof(int), cudaMemcpyHostToDevice);

    // HISTOGRAM CREATION PROCESS
    int blocksDim = 10;
    int threadsDim = hist_array_1D_size /blocksDim; 
    //printf("array_1D_size = %d\n",hist_array_1D_size);

    Histogram<<<blocksDim,threadsDim>>>(hist_array_1D_size,array_1D_size,hist_array_1D_cuda,array_1D_cuda);

    cudaMemcpy(hist_array_1D, hist_array_1D_cuda, hist_array_1D_size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    print_matrix(hist_array_1D,hist_array_1D_size);


    //printf("End of program");
    return 0;
}