#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;



int main()
{
    
    int i, throws = 99999, insideCircle = 0;
    double randX, randY, pi;
    int numthreads = omp_get_max_threads();
    int tid;
    uint32_t seed;

    double start,end;
    start = omp_get_wtime();

#pragma omp parallel private(tid,randX,randY,i,seed) num_threads(numthreads) reduction(+:insideCircle)
{ 
    tid = omp_get_thread_num();
    seed = (unsigned int) time(NULL);
    seed = (seed & 0xFFFFFFF0) | (tid + 1);
    srand(seed);
    
    #pragma omp for
    for (i = 0; i < throws; ++i) 
    {
      randX = (double) rand_r(&seed) / RAND_MAX;
      randY = (double) rand_r(&seed) / RAND_MAX;
      if (randX * randX + randY * randY < 1 + 1e-14) ++insideCircle;
    } 
}

    pi = 4.0 * insideCircle / throws;
    //pi = ((double)insideCircle/(double)(throws*numthreads))*4.0;
    cout << "pi = " << pi << endl;
  /*  
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by calculation: "
         << duration.count() << " microseconds" << endl;
 */
    end = omp_get_wtime();
    printf("Time of parallel pi monte carlo is %lf\n", end - start);        
         

return 0;
}
