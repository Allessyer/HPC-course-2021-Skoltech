#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;



void zero_init_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0.0;
        }
    }
}

void rand_init_matrix(double ** matrix, size_t N)
{
    double randvalue;
    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
	    randvalue = (double) rand();
            matrix[i][j] = randvalue / RAND_MAX;
        }
    }
}

double ** malloc_matrix(size_t N)
{
    double ** matrix = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; ++i)
    {   
        matrix[i] = (double *)malloc(N * sizeof(double));
    }
    
    return matrix;
}

void free_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; ++i)
    {   
        free(matrix[i]);
    }
    
    free(matrix);
}



int main()
{
    const size_t N = 1000; // size of an array

    double ** A, ** B, ** C; // matrices
    double start, end;

    printf("Starting:\n");

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);    

    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

//  matrix multiplication algorithm
    //auto start = high_resolution_clock::now();
    start = omp_get_wtime();

    for (int k = 0; k < N; k++)
    	for (int j = 0; j < N; j++)
    	    for (int i = 0; i < N; i++){
    	    	C[i][j] = A[i][k] * B[k][j];
    	    }    	    	
//
    end = omp_get_wtime(); 
    //auto stop = high_resolution_clock::now();
    //auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by KJI matmul: "
         << end - start << endl;


//  matrix multiplication algorithm
    //start = high_resolution_clock::now();
    start = omp_get_wtime();
    for (int i = 0; i < N; i++)
    	for (int j = 0; j < N; j++)
    	    for (int k = 0; k < N; k++){
    	    	C[i][j] = A[i][k] * B[k][j];
    	    } 

    end = omp_get_wtime();
    //stop = high_resolution_clock::now();
    //duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by IJK matmul: "
         << end - start << endl;


//  matrix multiplication algorithm
    //start = high_resolution_clock::now();
    start = omp_get_wtime();
    for (int j = 0; j < N; j++)
    	for (int i = 0; i < N; i++)
    	    for (int k = 0; k < N; k++){
    	    	C[i][j] = A[i][k] * B[k][j];
    	    }    


    end = omp_get_wtime();
    //stop = high_resolution_clock::now();
    //duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by JIK matmul: "
         << end - start << endl;


    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
