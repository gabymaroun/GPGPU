#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*typedef unsigned long long ttype;
ttype gettime(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (((ttype)ts.tv_sec*1e9) + ts.tv_nsec);
}*/


int main(int argc, char **argv)
{
  //if(argc!=2) {printf("Give the vector size as first parameter\n");exit(2);}
  int n = 1000000;
  printf("Vector size is %d\n",n);

 
  size_t bytes = n*sizeof(double);
  // Allocations on host
  //@TODO@ : complete here
  double *restrict a;
  double *restrict b;
  double *restrict c;

  a = (double*)malloc(bytes);
  b = (double*)malloc(bytes);
  c = (double*)malloc(bytes);
  
  //ttype tstart = gettime();
 
  int i ;

  for (i=0; i < n; ++i) {
    a[i] = i;
    b[i] = n-i;
    }

   #pragma acc data copyin(a[0:n],b[0:n]), copyout(c[0:n])
   #pragma acc parallel loop 
   for(i=0; i<n; i++) {
        c[i] = a[i] + b[i];
       //printf("vector a:%f\n",a[i]);
       //printf("vector b:%f\n",b[i]);
    }
   
  
   double sum=0.0;
   for (i=0; i<n;i++){
      sum += c[i];
   }
	//ttype tstop = gettime();
   //printf("time: %fs \n",(tstop-tstart)*1e-9); 
   
   printf("final result:%f\n",sum);


  // Free host memory
  //free(a); free(b); free(c);
  return 0;
}
