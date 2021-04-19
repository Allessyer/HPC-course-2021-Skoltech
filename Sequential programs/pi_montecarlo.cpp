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

srand(1);

auto start = high_resolution_clock::now();

for (i = 0; i < throws; ++i) {
  randX = rand() / (double) RAND_MAX;
  randY = rand() / (double) RAND_MAX;
  if (randX * randX + randY * randY < 1) ++insideCircle;
}

pi = 4.0 * insideCircle / throws;

auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);

cout << "Time of sequential pi monte carlo is "
         << duration.count() * 1e-6  << endl;
         


cout << "pi = " << pi << endl;


return 0;
}
