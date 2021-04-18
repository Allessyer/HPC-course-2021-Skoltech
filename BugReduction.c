#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

double dotprod(double * a, double * b, size_t N)
{
    int i, tid;
    double sum;

    tid = omp_get_thread_num();

#pragma omp parallel for reduction(+:sum)
    for (i = 0; i < N; ++i)
    {
        sum += a[i] * b[i];
        printf("tid = %d i = %d\n", tid, i);
    }
//    cout << "From dotprod function: " << sum << endl;

    return sum;
}

int main (int argc, char *argv[])
{
    const size_t N = 100;
    int i;
    double sum;
    double a[N], b[N];


    for (i = 0; i < N; ++i)
    {
        a[i] = b[i] = (double)i;
    }

    sum = 0.0;
    	
#pragma omp parallel shared(sum)
    sum = dotprod(a, b, N);

    printf("Sum = %f\n",sum);

    return 0;
}
