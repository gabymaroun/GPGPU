#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

// Initialize host vectors
void init(int *a, int *b, int n) {
  for (int i=0; i < n; ++i) {
    a[i] = i;
    b[i] = n-i;
  }
}

// Check result correctness
void check(int *c, int n) {
  int i = 0;
  while (i < n && c[i] == n) {
    ++i;
  }
  if (i == n)
    printf("Ok\n");
  else
    printf("Non ok\n");
}


// Cuda kernel
__global__ void add(int *a, int *b, int *c, int n) {
  //@TODO@ : complete kernel code
	int i = threadIdx.x + blockDim.x * blockIdx.x;
		if(i<n) 
			c[i] = a[i] + b[i];


}

int main(int argc, char **argv)
{
  if(argc!=2) {
	printf("Give the vector size as first parameter\n");
	exit(2);
	}

  int n = atoi(argv[1]);
  printf("Vector size is %d\n",n);
  // host pointers
  int *host_a, *host_b, *host_c;
  // Device pointers
  int *dev_a, *dev_b, *dev_c;

  // Allocations on host
  //@TODO@ : complete here
  host_a = (int *) malloc (n * sizeof(int));
  host_b = (int *) malloc (n * sizeof(int));
  host_c = (int *) malloc (n * sizeof(int));

  // Initialize vectors
  init(host_a,host_b,n);

  // Allocations on device
  //@TODO@ : complete here
   cudaMalloc(&dev_a, n * sizeof(int));
   cudaMalloc(&dev_b, n * sizeof(int));
   cudaMalloc(&dev_c, n * sizeof(int));

  // Copy from host to device
  //@TODO@ : complete here
   cudaMemcpy(dev_a, host_a, n * sizeof(int), cudaMemcpyHostToDevice );
   cudaMemcpy(dev_b, host_b, n * sizeof(int), cudaMemcpyHostToDevice );


  // Invoke kernel
  //@TODO@ : complete here
  dim3 DimGrid((n-1)/256+1,1,1);
  dim3 DimBlock(256,1,1);
  add<<< DimGrid, DimBlock>>>(dev_a, dev_b, dev_c, n);

  // Copy result from device to host
  //@TODO@ : complete here
  cudaMemcpy(host_c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost );


  // Check result
  check(host_c,n);

  // Free device memory
  cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
  // Free host memory
  free(host_a); free(host_b); free(host_c);
  return 0;
}
