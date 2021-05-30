
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>

using namespace std;

// CUDA KERNEL FUNCTIONS

__global__ void Hello(int *neighbours)
{
    //int globalidx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int globalidx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("blockIdx.x %d\n",blockIdx.x);
    printf("Hello from blockx = %d\t tx = %d\t ty=%d\t tz=%d\t gi=%d\n",blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z, globalidx);  
    
    for (int i = 0; i < 3; i++)
    {
        neighbours[i] = i;
    }
    
    for (int i = 0; i < 3; i++)
    {
        printf("neighbours[%d] = %d\n",i,neighbours[i]);
    }
    
}


__global__ void Bluring(int array_2D_size, int *array_1D_cuda, int *blured_1D_cuda, int *neighbours)
{
    int globalidx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Hello from blockx = %d\t tx = %d\t ty=%d\t tz=%d\t gi=%d\n",blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z, globalidx);  
    int N = array_2D_size;
    

    if(globalidx<N*N)
    {   
        if ((globalidx >= 0 && globalidx <= N-1) | (globalidx % N == 0) | (globalidx >= N*N-N) | (globalidx % N >= N -1 && globalidx % N <= N*N -1))
        {   
            blured_1D_cuda[globalidx]=array_1D_cuda[globalidx]; 
        } else 
        {   
            
            int top, bottom, left, right;
            top = -N + globalidx; // [i-1][j]
            bottom = N + globalidx; // [i+1][j]
            left = -1 + globalidx; // [i][j-1]
            right = 1 + globalidx; // [i][j+1]
            
            int cross1, cross2,cross3,cross4;
            cross1 = globalidx - N - 1; // [i-1][j-1]
            cross2 = globalidx - N + 1; // [i-1][j+1]
            cross3 = globalidx + N - 1;  // [i+1][j-1]
            cross4 = globalidx + N + 1;  // [i+1][j+1]
            
            /*
            if (globalidx == 7)
            {
                printf("top = %d\t bottom = %d\t left = %d\t right = %d\n", top, bottom, left, right);
                printf("cross1 = %d\t cross2 = %d\t cross3 = %d\t cross4 = %d\n", cross1, cross2, cross3, cross4);
            }
            */
            
            neighbours[0] = array_1D_cuda[top];
            neighbours[1] = array_1D_cuda[bottom];
            neighbours[2] = array_1D_cuda[left];
            neighbours[3] = array_1D_cuda[right];
            neighbours[4] = array_1D_cuda[cross1];
            neighbours[5] = array_1D_cuda[cross2];
            neighbours[6] = array_1D_cuda[cross3];
            neighbours[7] = array_1D_cuda[cross4];
            neighbours[8] = array_1D_cuda[globalidx];
          
            /*
            if (globalidx == 7)
            {
                for (int i = 0; i < 9; i++)
                {
                    printf("neighbours[%d] = %d\n",i,neighbours[i]);
                }
            }
            */
         
            // Сортировка массива пузырьком
            int size = 9;
            for (int i = 0; i < size - 1; i++)
            {
              for (int j = (size - 1); j > i; j--) // для всех элементов после i-ого
              {
                if (neighbours[j - 1] > neighbours[j]) // если текущий элемент меньше предыдущего
                {
                  int temp = neighbours[j - 1]; // меняем их местами
                  neighbours[j - 1] = neighbours[j];
                  neighbours[j] = temp;
                }
              }
            }
         
            /*
            if (globalidx == 7)
            {  
                printf("After bubble sort\n");
                for (int i = 0; i < 9; i++)
                {
                    printf("neighbours[%d] = %d\n",i,neighbours[i]);
                }
            }
            */
            
            blured_1D_cuda[globalidx] = neighbours[4];
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

/* Function to sort an array using insertion sort*/
void insertionSort(int arr[], int n)
{
    int i, key, j;
    for (i = 1; i < n; i++)
    {
        key = arr[i];
        j = i - 1;
  
        /* Move elements of arr[0..i-1], that are
        greater than key, to one position ahead
        of their current position */
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
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
 

    
    // ---------- CUDA ZONE -------------

    // printf("This is done by CPU before CUDA threads\n");

    // ARRAY TO DEVICE
    int *array_1D_cuda;
    cudaMalloc(&array_1D_cuda, sizeof(int)*array_1D_size);
    cudaMemcpy(array_1D_cuda, array_1D, array_1D_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // NEIGHBOURS ARRAY CREATION TO DEVICE
    int *neighbours;
    cudaMalloc(&neighbours, sizeof(int)*9);
    cudaDeviceSynchronize();

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
    // printf("array_1D_size = %d\n",array_1D_size );
    int array_2D_size = sqrt(array_1D_size);


    Bluring<<<blocksDim,threadsDim>>>(array_2D_size,array_1D_cuda, blured_1D_cuda, neighbours);

    cudaMemcpy(blured_1D, blured_1D_cuda, blured_1D_size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    convert_1d_to_2d(blured_1D_size, blured_1D);


    // printf("End of program");
    return 0;
}