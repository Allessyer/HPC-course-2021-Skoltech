
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;


// CUDA KERNEL FUNCTIONS

__global__ void Hello(void)
{
    int globalidx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x + 1;
    printf("Hello from tx = %d\t ty=%d\t tz=%d\t gi=%d\n", threadIdx.x, threadIdx.y, threadIdx.z, globalidx);
}

__global__ void Init(int n, float *d_a)
{
	
	int globalidx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if(globalidx<n) d_a[globalidx]=globalidx;
  printf("Array[%d] = %d from tx = %d\t ty=%d\t tz=%d\t gi=%d\n",globalidx,globalidx, threadIdx.x, threadIdx.y, threadIdx.z, globalidx);
	
}

__global__ void Initialization(int N, int N_2d, float *d_a)
{
	//  n = N * N
  
	int globalidx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if(globalidx<N) 
  {   
      if (globalidx >= 1 && globalidx <= N_2d-2) 
      {   
          printf("Hello from tx = %d\t ty=%d\t tz=%d\t gi=%d\n", threadIdx.x, threadIdx.y, threadIdx.z, globalidx);
          d_a[globalidx]=1.0;
      } else
      {
          d_a[globalidx]=0.0;
      }
      
  }
  printf("from tx = %d\t ty=%d\t tz=%d\t gi=%d\n", threadIdx.x, threadIdx.y, threadIdx.z, globalidx);
  
}

__global__ void Laplace(int N, float *T, float *d_res)
{
    int globalidx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;  
    if(globalidx<N*N)
    {
        if ((globalidx >= 0 && globalidx <= N-1) | (globalidx % N == 0) | (globalidx >= N*N-N) | (globalidx % N >= N -1 && globalidx % N <= N*N -1))
        {   
            //printf("Hello from tx = %d\t ty=%d\t tz=%d\t gi=%d\n", threadIdx.x, threadIdx.y, threadIdx.z, globalidx);  
            d_res[globalidx]=T[globalidx];
         
        } else 
        {   
            int top, bottom, left, right;
            top = -N + globalidx;
            bottom = N + globalidx;
            left = -1 + globalidx;
            right = 1 + globalidx;
            d_res[globalidx]=0.25 * (T[top] + T[bottom] + T[left] + T[right]);
        }
        
    } 
    

}



// CPU FUNCTION

void print_matrix(float * matrix, int N)
{
    for (int j = 0; j < N; j++)
    {
        cout << matrix[j] << " ";
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

float ** malloc_matrix(size_t N)
{
    float ** matrix = (float **)malloc(N * sizeof(float *));
    
    for (int i = 0; i < N; ++i)
    {   
        matrix[i] = (float *)malloc(N * sizeof(float));
    }
    
    return matrix;
}


int main(void)
{
    printf("This is done by CPU before CUDA threads\n");
    Hello<<<1,4>>>();
    cudaDeviceSynchronize();
 
    int N = 20;
    
    int n = N*N;
    int MAX_ITER = 500;

    printf("Now let's work with arrays\n");
    
    float *h_a = (float*)malloc(sizeof(float)*n);
    float *h_init = (float*)malloc(sizeof(float)*n);
 
    float *d_a;
    float *d_temp1,*d_temp2;
    float *d_res;

    cudaMalloc(&d_a, sizeof(float)*n);
    cudaMalloc(&d_temp1, sizeof(float)*n);
    cudaMalloc(&d_temp2, sizeof(float)*n);
    cudaMalloc(&d_res, sizeof(float)*n); 

    // 1 block and N threads
    
    Initialization<<<1,n>>>(n,N, d_a);
 
    cudaMemcpy(h_init, d_a, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("From CPU print INIT aray\n");
    print_matrix(h_init,n);
 

    float ** h_init_2D;
    h_init_2D = malloc_matrix(N);
 
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_init_2D[i][j] = h_init[i*N + j];
        }
    }

    printf("From CPU print INIT aray\n");
    print_matrix(h_init_2D,N);
    
    Laplace<<<1,n>>>(N, d_a, d_temp1);
    cudaMemcpy(h_a, d_temp1, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("From CPU print aray\n");
    print_matrix(h_a,n);
 
    float ** h_a_2D;
    h_a_2D = malloc_matrix(N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_a_2D[i][j] = h_a[i*N + j];
        }
    }

    
    int k = 0;
    while(k<MAX_ITER) {
        Laplace<<<1,n>>>(N, d_temp1, d_temp2);   // update T1 using data stored in T2
        /*
        cudaMemcpy(h_a, d_temp1, n*sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        printf("From CPU print aray\n");
        print_matrix(h_a,n);

        //float ** h_a_2D;
        //h_a_2D = malloc_matrix(N);
    
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                h_a_2D[i][j] = h_a[i*N + j];
            }
        }

        printf("From CPU print aray\n");
        print_matrix(h_a_2D,N);
        */
        Laplace<<<1,n>>>(N, d_temp2, d_temp1);   // update T2 using data stored in T1
        k+=2;
    }
      
    cudaMemcpy(h_a, d_temp1, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("From CPU print aray\n");
    print_matrix(h_a,n);
    

    //float ** h_a_2D;
    //h_a_2D = malloc_matrix(N);
 
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_a_2D[i][j] = h_a[i*N + j];
        }
    }

    printf("From CPU print aray\n");
    print_matrix(h_a_2D,N);
    
    

    ofstream myfile;
    myfile.open ("T_cuda_array.txt");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            myfile << h_a_2D[i][j] << " ";
        }
        myfile << endl;
    }
    myfile.close();
    
    
    
    

    return 0;
}
