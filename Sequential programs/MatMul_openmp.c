#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <omp.h>
using namespace std::chrono;
using namespace std;

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
    int i,j,k;
//    clock_t start, end;   
 
    double ** A, ** B, ** C; // matrices

    printf("Starting:\n");

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);    

    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

/*    cout << "Matrix A" << endl;
    for (int i = 0; i < N; i++)
    {
	for (int j = 0; j < N; j++)
	{
		cout << A[i][j] << " ";
	}
	cout << endl;
    }
*/
//    start = clock();
    auto start = high_resolution_clock::now();
//  matrix multiplication algorithm
#pragma omp parallel shared(A,B,C) private(i,j,k)
    {
        #pragma omp for schedule(static) 

    for (int k = 0; k < N; k++)
    	for (int j = 0; j < N; j++)
    	    for (int i = 0; i < N; i++){
    	    	C[i][j] = A[i][k] * B[k][j];
    	    }
    }    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by parallel KJI matmul: "
         << duration.count() << " microseconds" << endl;
//

   // end = clock();

  //  printf("Time elapsed paralleled (kji): %f seconds.\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = high_resolution_clock::now();
    
//  matrix multiplication algorithm OpenMP

#pragma omp parallel shared(A,B,C) private(i,j,k)
    {
        #pragma omp for schedule(static) 
    	    for (int i = 0; i < N; i++)
    		for (int j = 0; j < N; j++)
    	    	    for (int k = 0; k < N; k++){
    	    		C[i][j] = A[i][k] * B[k][j];
    	    	    }    	    	
    }

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by parallel IGK matmul: "
         << duration.count() << " microseconds" << endl;
//


    start = high_resolution_clock::now();
//  matrix multiplication algorithm
#pragma omp parallel shared(A,B,C) private(i,j,k)
    {
        #pragma omp for schedule(static) 
    for (int j = 0; j < N; j++)
    	for (int i = 0; i < N; i++)
    	    for (int k = 0; k < N; k++){
    	    	C[i][j] = A[i][k] * B[k][j];
    	    }    
    }    
//

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by parallel JIK matmul: "
         << duration.count() << " microseconds" << endl;



    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
