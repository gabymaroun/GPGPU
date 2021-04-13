#include <stdio.h>
#include <cuda.h>
#include <time.h>

#include "matmul_utils.hpp"


int main(int argc, char **argv)
{
  if(argc!=4) {printf("Usage : %s [nb of rows for A] [nb of cols for A] [nb of cols for B]\n", argv[0]);exit(2);}
  //initilize a pseudo-random number generator
  srand(time(0));

  // Read given dimensions
  int numARows = atoi(argv[1]);
  int numAColumns = atoi(argv[2]);
  int numBColumns  = atoi(argv[3]);
  // Compute the remaining dimensions for given ones
  int numBRows = numAColumns;
  int numCRows = numARows;
  int numCColumns = numBColumns;
  printf("Matrix multiplication dimensions: [%d;%d] = [%d;%d] x [%d;%d]\n",
         numCRows, numCColumns, numARows, numAColumns, numBRows, numBColumns);
  // host pointers
  float * a, * b, * c;

  // Allocations on host
  a = (float *)calloc(numARows*numAColumns, sizeof(float));
  b = (float *)calloc(numBRows*numBColumns, sizeof(float));
  c = (float *)calloc(numCRows*numCColumns, sizeof(float));

  // Initialize vectors
  init(a,b,numARows, numAColumns, numBRows, numBColumns);

  // Matrix mutliplication
int na=numAColumns *numARows;
int nb=numBColumns *numBRows;
int nc=numCColumns *numCRows;

#pragma acc parallel loop  collapse(2) gang copyout(c[0:nc]), copyin(a[0:na], b[0:nb])
  for (int i=0; i < numARows; i++) {
    for (int j=0; j < numBColumns; j++) {
      float cij = 0.0;
      #pragma acc loop vector
      for (int k=0; k < numAColumns; k++) {
        cij += a[i*numAColumns+k]*b[k*numBColumns+j];
      }
      c[i*numAColumns+j] = cij;
    }
  }

  // Check result
  check(a,b,c,numARows, numAColumns, numBRows, numBColumns);
  
  // Free host memory
  free(a); free(b); free(c);
  return 0;
}

