#include <stdio.h>
#include <cuda.h>


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
	if(i < n) {
		c[i]=a[i]+b[i];
	}
}

int main(int argc, char **argv)
{	

  if(argc<2) {printf("Give the vector size as first parameter\n");;exit(2);}


  int n = atoi(argv[1]);
  printf("Vector size is %d\n",n);

  // host pointers
  int *host_a, *host_b, *host_c;
  int STREAM_NB=4;
  int STREAM_SIZE=512;


  // Allocations on host
  //@TODO@ : 
	cudaHostAlloc((void **) &host_a, n*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **) &host_b, n*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **) &host_c, n*sizeof(int), cudaHostAllocDefault);


  // Initialize vectors
  init(host_a,host_b,n);


  cudaStream_t streams[STREAM_NB];
  int *d_A[STREAM_NB];
  int *d_B[STREAM_NB];
  int *d_C[STREAM_NB];


   for(int i=0;i<STREAM_NB;i++)
  {
    cudaStreamCreate(&streams[i]); 
    cudaMalloc((void**)&d_A[i],STREAM_SIZE*sizeof(int));
    cudaMalloc((void**)&d_B[i],STREAM_SIZE*sizeof(int));
    cudaMalloc((void**)&d_C[i],STREAM_SIZE*sizeof(int));
  }


for (int i=0; i<n; i+=STREAM_SIZE*STREAM_NB) 
{
  for(int j=0;j<STREAM_NB;j++)
 {
     cudaMemcpyAsync(d_A[j], host_a+i+STREAM_SIZE*j,STREAM_SIZE*sizeof(int),cudaMemcpyHostToDevice,streams[j]);
     
     cudaMemcpyAsync(d_B[j], host_b+i+STREAM_SIZE*j, STREAM_SIZE*sizeof(int),cudaMemcpyHostToDevice,streams[j]);

     add<<<STREAM_SIZE/256, 256, 0, streams[j]>>>(d_A[j], d_B[j],d_C[j],STREAM_SIZE);

     cudaMemcpyAsync(host_c+i+STREAM_SIZE*j,d_C[j], STREAM_SIZE*sizeof(int),cudaMemcpyDeviceToHost,streams[j]);
 }
}
  cudaDeviceSynchronize();


  // Check result
  check(host_c,n);


  // Free device memory in a loop :
  for(int i=0;i<STREAM_NB;i++)
{
  cudaFree(d_A[i]); 
  cudaFree(d_B[i]); 
  cudaFree(d_C[i]);
}


 cudaFree(host_a);
 cudaFree(host_b); 
 cudaFree(host_b);
  return 0;
}
