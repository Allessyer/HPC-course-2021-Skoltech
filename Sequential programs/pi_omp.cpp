#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <omp.h>
using namespace std;
using namespace std::chrono;


const size_t N = 1e8;
const int N_THREADS = omp_get_max_threads();
//unsigned int seeds[N_THREADS];

std::vector<unsigned int> seeds(N_THREADS);


void seedThreads() {
    int my_thread_id;
    unsigned int seed;
    #pragma omp parallel private (seed, my_thread_id)
    {
        my_thread_id = omp_get_thread_num();
        
        //create seed on thread using current time
        unsigned int seed = (unsigned) time(NULL);
        
        //munge the seed using our thread number so that each thread has its
        //own unique seed, therefore ensuring it will generate a different set of numbers
        seeds[my_thread_id] = (seed & 0xFFFFFFF0) | (my_thread_id + 1);
        
        printf("Thread %d has seed %u\n", my_thread_id, seeds[my_thread_id]);
    }
    
}


int main()
{
    
    int i, throws = 10e6, insideCircle = 0;
    double randX, randY, pi;
    int tid;
    unsigned int  seed;
    omp_set_num_threads(N_THREADS);
    seedThreads();

    double start,end;
    
    printf("N_THREADS = %d\n", N_THREADS);
    start = omp_get_wtime();

#pragma omp parallel default(none) \
                     shared(seeds,throws) \
                     private(tid,randX,randY,seed) \
                     num_threads(N_THREADS) \
                     reduction(+:insideCircle)
{ 
    tid = omp_get_thread_num();
    seed = seeds[tid];
    srand(seed);
    
    #pragma omp for
    for (i = 0; i < throws; ++i) 
    {
      randX = rand_r(&seed) / (double) RAND_MAX;
      randY = rand_r(&seed) /(double) RAND_MAX;
      if (randX * randX + randY * randY < 1) {
           insideCircle += 1;
      }
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
