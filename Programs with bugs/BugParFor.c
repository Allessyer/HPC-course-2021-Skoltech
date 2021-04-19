#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
    const size_t N = 5;
    const size_t chunk = 3;

    int i, tid;
    float a[N], b[N], c[N];
    double start, end;

    for (i = 0; i < N; ++i)
    {
        a[i] = b[i] = (float)i;
    }
    
    printf("Sequential part: \n\n");
    start = omp_get_wtime();
    for (i = 0; i < N; ++i)
    {
        c[i] = a[i] + b[i];
        printf("c[%d] = %f\n", i, c[i]);
    }
    end = omp_get_wtime();
    printf("Sequential Time: %f\n",end - start);
    
    printf("\n");
  
    printf("Parallel part: \n");
    printf("N = %ld, chunk = %ld\n\n",N,chunk);
    start = omp_get_wtime();

#pragma omp parallel for default(none) shared(a,b,c,N,chunk) private(tid) schedule(static,chunk)
for (i = 0; i < N; ++i)
{
    tid = omp_get_thread_num();
    c[i] = a[i] + b[i];
    printf("tid = %d, c[%d] = %f\n", tid, i, c[i]);
}

    end = omp_get_wtime();
    printf("Parallel Time: %f\n",end - start);

    return 0;
}
