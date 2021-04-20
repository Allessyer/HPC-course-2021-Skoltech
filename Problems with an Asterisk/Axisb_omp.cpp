#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>



void gauss_seidel_OMP(float *A, float *b, float *x, int N) {

    double *S = new double[N*N];
    double *S_T = new double[N*N];
    double *diagonal = new double[N];
    double *x_T = new double[N];

    int ITERATION_LIMIT = 1000;
    double rtol = 1e-7;
    int CHUNK = 256;
    

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double c;
            if (i != j) 
            {
                c = -A[i*N + j]/A[i*N + i];
            } else 
            {
                c = 0.;
            }
            S[i*N + j] = c;
            S_T[j*N + i] = c;
        }
    }

    for (int i = 0; i < N; i++) {
        diagonal[i] = b[i]/A[i*N + i];
    }

    for (int i = 0; i < N; i++) {
        x[i] = 0.;
    }

    for (int it = 0; it < ITERATION_LIMIT; it++) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double acc = 0.;
            for (int j = i; j < N; j++) {
                acc += S[i*N + j] * x[j];
            }
            x_T[i] = acc + diagonal[i];
        }
        
        double error = 0.;
        float x_current, x_previous;
        for (int i = 0; i < N; i+=CHUNK) {
            int M = i + CHUNK > N ? N : i + CHUNK;
            for (int j = i; j < M; j++) {
                x_previous = x[j];
                x_current = x_T[j];
                for (int k = j; k < M; k++) {
                    x_T[k] += S_T[j*N + k] * x_current;
                }
                error += fabs(x_current - x_previous);
                x[j] = x_current;
            }

            #pragma omp parallel for
            for (int k = M; k < N; k++) {
                float acc = 0.;
                for (int j = i; j < M; j++) {
                    acc += S[k*N + j] * x[j]; 
                }
                x_T[k] += acc;
            }
        }

        if (error< rtol) break;
    }

    free(S);
    free(S_T);
    free(x_T);
    free(diagonal);
    
}



int main() {
    int N = 2000;
    float *A = new float[N*N];
    float *b = new float[N];
    float *x = new float[N];

    unsigned int seed = 0;

    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            double a;
            if (i!=j) 
            { 
                a = -0.5 + (float)rand_r(&seed) / RAND_MAX;
            } else 
            {
                a = 100. + (float)rand_r(&seed) / RAND_MAX;
            }
            A[i*N + j] = a;
            A[j*N + i] = a;
        }
    }

    for (int i = 0; i < N; i++) {
        b[i] = (float)rand_r(&seed) / RAND_MAX;
    }

    double start, end, error;
    start = omp_get_wtime();
    gauss_seidel_OMP(A, b, x, N);
    end = omp_get_wtime();

    error = 0.;
    for (int i = 0; i < N; i++) {
        double err = 0.;
        for (int j = 0; j < N; j++) {
            err += A[i*N + j] * x[j];
        }
        err -= b[i];
        error += fabs(err);
    }

    printf("Parallel Gaussian-Seidel algorithm: \n\n");
    printf("Time is %f seconds, error = %.5f\n", end - start, error);

    return 0;
}
