#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <chrono>
#include <math.h>
#include<iomanip>
#include<cmath>
#include <omp.h>
#include <time.h>
#define _USE_MATH_DEFINES

using namespace std;
using namespace std::chrono;



void data_generation(double *x, double *y, int N) {

    double a = 1.47;
    double b = 3.0;
    
    double* noise = new double[N];
    unsigned int seed;
    
    for(int i = 0; i < N; ++i) {
         seed = (unsigned int)rand();
         double r1 = (double)rand_r(&seed) / RAND_MAX;
         double r2 = (double)rand_r(&seed) / RAND_MAX;

         double r = sqrt(log(1/(1 - r1 + 1e-20)));

         noise[i] = 1.0 * r * cos(2*M_PI*r2);
    }
     
   
    for(int i = 0; i < N; ++i) {
        seed = (unsigned int)rand();
        x[i] = 2*((double)rand_r(&seed) / RAND_MAX) - 1;
    }
    
    
    for(int i = 0; i < N; ++i)
    {
    	y[i] = a*x[i] + b + noise[i];
    }
    
    cout << "Real parameters: a = " << a << setw(19) << " b = " << b << endl;
    
}

int main() 
{   
    // generation of data
    
    int N = 10e8;
    double* x = new double[N];
    double* y = new double[N];
    
    data_generation(x,y,N);
    
    // Least Squered fitting
    
    double a_theor, b_theor;
    double x_mean=0,x2_mean=0,y_mean=0,xy_mean=0;
    int numthreads = omp_get_max_threads();   
    
    double start, end;
    
    start = omp_get_wtime();             
    
    #pragma omp parallel reduction(+:x_mean,y_mean,x2_mean,xy_mean) num_threads(numthreads)
{
    #pragma omp for schedule(static) 
    for (int i=0; i < N; i++)
    {
        x_mean += x[i];                        //calculate sigma(xi)
        y_mean += y[i];                        //calculate sigma(yi)
        x2_mean += pow(x[i],2);                //calculate sigma(x^2i)
        xy_mean += x[i]*y[i];                    //calculate sigma(xi*yi)
    }
    
}
    
    x_mean /= N;
    y_mean /= N;
    x2_mean /= N;
    xy_mean /= N;
    
    end = omp_get_wtime();
    
    
    a_theor = (xy_mean - x_mean*y_mean) / (x2_mean - x_mean*x_mean);            //calculate slope
    
    b_theor= y_mean - a_theor*x_mean;            //calculate intercept
    cout << "Theor parameters: a = " << a_theor<< setw(19-4)<< "  b = " << b_theor << endl;
    cout << "Time taken for regression: " << end - start << endl;
        
return 0;
}
