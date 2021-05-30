
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


__global__ void Bluring(int array_2D_size, int summary, int *array_1D_cuda, int *blured_1D_cuda, int *cu_kernel_1d)
{
    int globalidx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Hello from blockx = %d\t tx = %d\t ty=%d\t tz=%d\t gi=%d\n",blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z, globalidx);  
    int N = array_2D_size;
    

    if(globalidx<N*N)
    {   
        if ((globalidx >= 0 && globalidx <= N-1) | (globalidx % N == 0) | (globalidx >= N*N-N) | (globalidx % N >= N -1 && globalidx % N <= N*N -1))
        {   
            blured_1D_cuda[globalidx]=array_1D_cuda[globalidx]; 
         /*
            if (globalidx == 0)
            {
                printf("blured_1D_cuda[%d] = %d\n",globalidx,blured_1D_cuda[globalidx]);
            }        
         */

        } else 
        {   
            int top, bottom, left, right;
            top = -N + globalidx; // [i-1][j]
            bottom = N + globalidx; // [i+1][j]
            left = -1 + globalidx; // [i][j-1]
            right = 1 + globalidx; // [i][j+1]
            
            /*
            if (globalidx == 6)
            {
                printf("top = %d\t bottom = %d\t left = %d\t right = %d\n", top,bottom,left,right);
            }
         */
            
            int cross1, cross2,cross3,cross4;
            cross1 = globalidx - N - 1; // [i-1][j-1]
            cross2 = globalidx - N + 1; // [i-1][j+1]
            cross3 = globalidx + N - 1;  // [i+1][j-1]
            cross4 = globalidx + N + 1;  // [i+1][j+1]
         
         /*
            if (globalidx == 6)
            {
                printf("cross1 = %d\t cross2 = %d\t cross3 = %d\t cross4 = %d\n", cross1, cross2,cross3,cross4);
            }
         */
        
            int T1,T2,T3,T4,T5,T6,T7,T8,T9;
            int temp1,temp2,temp3;
        
            T1 = array_1D_cuda[cross1] * cu_kernel_1d[0*3+0];
            T2 = array_1D_cuda[top] * cu_kernel_1d[0*3+1];
            T3 = array_1D_cuda[cross2] * cu_kernel_1d[0*3+2];
            temp1 = T1 + T2 + T3;
         /*
            if (globalidx == 6)
            {
                printf("T1 = %d\t T2 = %d\t T3 = %d\t temp1 = %d\n", T1, T2,T3,temp1);
            }
         */
            
            T4 = array_1D_cuda[left] * cu_kernel_1d[1*3+0];
            T5 = array_1D_cuda[globalidx] * cu_kernel_1d[1*3+1];
            T6 = array_1D_cuda[right] * cu_kernel_1d[1*3+2];
            temp2 = T4 + T5 + T6;
         /*
            if (globalidx == 6)
            {
                printf("T4 = %d\t T5 = %d\t T6 = %d\t temp1 = %d\n", T4, T5,T6,temp2);
            }
          */
            T7 = array_1D_cuda[cross3] * cu_kernel_1d[2*3+0];
            T8 = array_1D_cuda[bottom] * cu_kernel_1d[2*3+1];
            T9 = array_1D_cuda[cross4] * cu_kernel_1d[2*3+2];
            temp3 = T7 + T8 + T9;
            
        /*
            if (globalidx == 6)
            {
                printf("T7 = %d\t T8 = %d\t T9 = %d\t temp3 = %d\n", T7, T8,T9,temp3);
            }
         */
        
            blured_1D_cuda[globalidx] = temp1+temp2+temp3;
        }
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

    //convert_1d_to_2d(array_1D_size, array_1D);

    // CREATION OF CONVOLUTION

    int **kernel;
    kernel = malloc_matrix(3);
    for (int i = 0; i < 3; i++)
    {
        kernel[i][0] = 1;
        kernel[i][1] = 2;
        kernel[i][2] = 1;
        if (i == 1)
        {
            kernel[i][0] *= 2;
            kernel[i][1] *= 2;
            kernel[i][2] *= 2;
        }
    }
    
    //print_matrix(kernel,3);

    // 2D KERNEL TO 1D KERNEL

    int *kernel_1d = (int*)malloc(sizeof(int)*3*3);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            kernel_1d[i*3 + j] = kernel[i][j];
        }
    }
    
    //print_matrix(kernel_1d,3*3);

    // SUM of 1D KERNEL

    int summary = 0;
    for (int j = 0; j < 3*3; j++)
    {
        summary += kernel_1d[j];
    }
    //printf("summary = %d\n", summary);
    //printf("This is done by CPU before CUDA threads\n");


    // ---------- CUDA ZONE -------------
    
    //printf("Start to allocate summary in CUDA\n");

    // CONV KERNEL kernel_1d TO DEVICE
    int *cu_kernel_1d;
    int kernel_1d_size = 9;
    cudaMalloc(&cu_kernel_1d, sizeof(int)*kernel_1d_size);
    cudaMemcpy(cu_kernel_1d, kernel_1d, kernel_1d_size * sizeof(int), cudaMemcpyHostToDevice);
 
    // ARRAY TO DEVICE
    int *array_1D_cuda;
    cudaMalloc(&array_1D_cuda, sizeof(int)*array_1D_size);
    cudaMemcpy(array_1D_cuda, array_1D, array_1D_size * sizeof(int), cudaMemcpyHostToDevice);

    // BLURED ARRAY ON HOST
    int blured_1D_size = array_1D_size;
    int *blured_1D = (int*)malloc(sizeof(int)*blured_1D_size);

    // BLURED ARRAY ON DEVICE
    int *blured_1D_cuda;
    cudaMalloc(&blured_1D_cuda, sizeof(int)*blured_1D_size);
    cudaMemcpy(blured_1D_cuda, blured_1D, blured_1D_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // BLURING PROCESS
    int blocksDim = 10;
    int threadsDim = array_1D_size /blocksDim; 
    //printf("array_1D_size = %d\n",array_1D_size );
    int array_2D_size = sqrt(array_1D_size);


    Bluring<<<blocksDim,threadsDim>>>(array_2D_size, summary, array_1D_cuda, blured_1D_cuda,cu_kernel_1d);

    cudaMemcpy(blured_1D, blured_1D_cuda, blured_1D_size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //convert_1d_to_2d(blured_1D_size, blured_1D);

    // SCALE BLURED ARRAY
    int *scaled_blured_1D = (int*)malloc(sizeof(int)*blured_1D_size);
    for (int i = 0; i < blured_1D_size; i++)
    {
        scaled_blured_1D[i] = blured_1D[i] /summary;
        //cout << scaled_blured_1D[i] << endl;
    }
    
    convert_1d_to_2d(blured_1D_size, scaled_blured_1D);

    //printf("End of program");
    return 0;
}