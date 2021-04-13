#include <cstdlib>
#include <iostream>

#include "matmul_utils.hpp"

// Initialize host matrices
void init(float *a, float *b, int nRA, int nCA, int nRB, int nCB) {
  for (int j=0; j < nCA; j++) {
    for (int i=0; i < nRA; i++) {
      a[i*nCA+j] = rand()/((float)RAND_MAX);
    }
  }
  for (int j=0; j < nCB; j++) {
    for (int i=0; i < nRB; i++) {
      b[i*nCB+j] = rand()/((float)RAND_MAX);
    }
  }
}

// Check result correctness
void check(float *a, float *b, float *c, int nRA, int nCA, int nRB, int nCB) {
  int isok=1;
  for (int j=0; j < nCB && isok; j++) {
    for (int i=0; i < nRA && isok; i++) {
      float cij = 0.0;
      for (int k=0; k < nCA; k++) {
        cij += a[i*nCA+k]*b[k*nCB+j];
      }
      isok = isok && (abs(cij-c[i*nCB+j])<1e-10);
    }
  }
  if (isok)
    std::cout << "Ok" << std::endl;
  else
    std::cout << "NON Ok !" << std::endl;
}
