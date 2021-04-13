#include <stdio.h>
#include <cuda.h>
#include <time.h>

#include "matmul_utils.hpp"
#define TILE_WIDTH 16

// Cuda kernel

__global__ void dgemm(float *A, float *B, float *C,
                      int numARows, int numAColumns, int numBRows, int numBColumns)  {

   
   __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
   __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

   int bx = blockIdx.x;  int by = blockIdx.y;
   int tx = threadIdx.x; int ty = threadIdx.y;
  

    int Row= by * TILE_WIDTH + ty;
    int Col= bx* TILE_WIDTH+ tx;

    float Pvalue =0;

    for(int k=0; k< ceil((float)numAColumns/TILE_WIDTH); k++) {    
       if ( Row < numARows && k*TILE_WIDTH+tx < numAColumns){
			ds_A[ty][tx] = A[ Row * numAColumns+ k*TILE_WIDTH+tx];
			}
       else{
           ds_A[ty][tx]=0;
	   }
       if ( k*TILE_WIDTH+ty< numBRows && Col < numBColumns){
			ds_B[ty][tx] = B[ ( k * TILE_WIDTH + ty) * numBColumns + Col];
			}
       else{
           ds_B[ty][tx] =0; 
	   }
		__syncthreads(); 
    
      for(int i=0; i< TILE_WIDTH && i < numAColumns && i< numBRows; i++){
		  Pvalue += ds_A[ty][i] * ds_B[i][tx];
		__syncthreads();
	  }
	}
      
     if (Row < numARows && Col < numBColumns ) 
        C[Row*numBColumns + Col ] = Pvalue;
      
}

int main(int argc, char **argv)
{
  if(argc!=4) {printf("Usage : %s [nb of rows for A] [nb of cols for A] [nb of cols for B]\n", argv[0]);exit(2);}
  //initilize a pseudo-random number generator
  srand(time(0));

 
  int numARows, numAColumns,numBRows, numBColumns,numCRows, numCColumns;
  // Read given dimensions
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBColumns  = atoi(argv[3]);
  // Compute the remaining dimensions for given ones
  //@TODO@
  numBRows = numAColumns;
  //@TODO@ 
  numCRows =  numARows; 
  //@TODO@
  numCColumns = numBColumns ;

  printf("Matrix multiplication dimensions: [%d;%d] = [%d;%d] x [%d;%d]\n",
         numCRows, numCColumns, numARows, numAColumns, numBRows, numBColumns);
  // host pointers
  float *host_a, *host_b, *host_c;
  // Device pointers
  float *dev_a, *dev_b, *dev_c;

  // Allocations on host
  host_a = (float *)calloc(numARows*numAColumns, sizeof(float));
  host_b = (float *)calloc(numBRows*numBColumns, sizeof(float));
  host_c = (float *)calloc(numCRows*numCColumns, sizeof(float));

  // Initialize vectors
  init(host_a,host_b,numARows, numAColumns, numBRows, numBColumns);

  // Allocations on device
  // @TODO@ : complete device allocations
   cudaMalloc(&dev_a, numARows*numAColumns *sizeof(float));
   cudaMalloc(&dev_b, numBRows*numBColumns *sizeof(float));
   cudaMalloc(&dev_c, numCRows*numCColumns *sizeof(float));

  // Copy from host to device
  // @TODO@ : complete copy from host to device
   cudaMemcpy(dev_a,host_a, numARows*numAColumns*sizeof(float),cudaMemcpyHostToDevice);
   cudaMemcpy(dev_b, host_b, numBRows*numBColumns*sizeof(float),cudaMemcpyHostToDevice);


  // Invoke kernel
  // @TODO@ : complete compute grid and block dim
   dim3 DimGrid((numCColumns-1)/TILE_WIDTH+1 , (numCRows-1)/TILE_WIDTH+1,1);
   dim3 DimBlock (TILE_WIDTH,TILE_WIDTH,1);

  // Initialize C device data
  cudaMemset(dev_c, 0, numCRows * numCColumns * sizeof(float));

  // Call the kernel
  // @TODO@ : complete to call the kernel
   dgemm<<< DimGrid, DimBlock  >>>(dev_a, dev_b, dev_c, numARows, numAColumns, numBRows, numBColumns);

  // Copy result from device to host
  // @TODO@ : complete copy from device to host
  cudaMemcpy(host_c, dev_c, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);

  // Check result
  check(host_a,host_b,host_c,numARows, numAColumns, numBRows, numBColumns);

  // Free device memory
  // @TODO@ : complete to deallocate memory
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    cudaFree(host_a); cudaFree(host_b); cudaFree(host_c);


  return 0;
}
