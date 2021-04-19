#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

double dotprod(double * a, double * b, size_t N)
{
    int i, tid;
    double sum;

    //tid = omp_get_thread_num();

#pragma omp parallel for private(tid) reduction(+:sum) 
    for (i = 0; i < N; ++i)
    {
        tid = omp_get_thread_num();
        sum += a[i] * b[i];
        printf("tid = %d i = %d\n", tid, i);
    }

    return sum;
}

int main (int argc, char *argv[])
{
    const size_t N = 20;
    int i;
    double sum;
    double a[N], b[N];
    
    double start, end;
    
    cout << "N = " << N << endl;
    
    for (i = 0; i < N; ++i)
    {
        a[i] = b[i] = (double)i + 1;
    }
    

    sum = 0.0;
    
    cout << "Sequential part: " << endl;
    
    start = omp_get_wtime();
    for (i = 0; i < N; ++i)
    {
    	sum += a[i] * b[i];
    }
    
    cout << "Sequential SUM = " << sum << endl;
    end = omp_get_wtime();
    
    cout << "Sequential Time: " << end - start << endl;
    cout << endl;
    
    sum = 0.0;
    
    cout << "Parallel part: " << endl;
    
    start = omp_get_wtime();	
    //#pragma omp parallel
    sum = dotprod(a, b, N);
    
    cout << "Parallel SUM = " << sum << endl;
    end = omp_get_wtime();
    cout << "Parallel Time: " << end - start << endl;
    

    return 0;
}
