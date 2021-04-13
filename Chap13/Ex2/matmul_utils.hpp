#ifndef MATMUL_UTILS_H
#define MATMUL_UTILS_H


// Initialize host matrices
void init(float *a, float *b, int nRA, int nCA, int nRB, int nCB);
// Check result correctness
void check(float *a, float *b, float *c, int nRA, int nCA, int nRB, int nCB);


#endif /* MATMUL_UTILS_H */
