#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>

void gauss_seidel_single(float *A, float *b, float *x, int N) {
	
    double *S = new double[N*N];
    double *diagonal = new double[N];
    double *allclose = new double[N];
    
    
    int ITERATION_LIMIT = 1000;
    double rtol = 1e-7;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j) 
            {
                S[i*N + j] = -A[i*N + j]/A[i*N + i];
            } else
            {
                S[i*N + j] = 0.;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        diagonal[i] = b[i]/A[i*N + i];
    }

    // x_new = np.zeros_like(x) 
    for (int i = 0; i < N; i++) {
        x[i] = 0.;
    }

    for (int it = 0; it < ITERATION_LIMIT; it++) {
        for (int i = 0; i < N; i++) {
            double new_x = 0.;
            for (int j = 0; j < N; j++) {
                new_x += S[i*N + j] * x[j];
            }
            new_x += diagonal[i];
            allclose[i] = new_x - x[i];
            x[i] = new_x;
        }

        double error = 0.;
        for (int i = 0; i < N; i++) {
            error += fabs(allclose[i]);
        }
        if (error < rtol) break;
    }

    free(S);
    free(diagonal);
    free(allclose);
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
    gauss_seidel_single(A, b, x, N);
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

    printf("Sequential Gaussian-Seidel algorithm: \n\n");
    printf("Time is %f seconds, error = %.5f\n", end - start, error);

    return 0;
}
