#include <iostream>
#include <cstdlib>
#include <string>
//#include <time.h>

#include "img_utils.hpp"
/*typedef unsigned long long ttype;
ttype gettime(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (((ttype)ts.tv_sec*1e9) + ts.tv_nsec);
}*/


//#define maskCols 5
//#define maskRows 5
//#define filter_width  5
// #define Mask_radius Mask_width/2
#define TILE_WIDTH 32
// #define w (TILE_WIDTH + filter_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

using namespace std;

//@TODO@ : Write the kernel here

__global__ void convolution_2D_tiled_kernel(float *I, const float* __restrict__ M, float *P, int channels, int width, int height, int filter_width) {

   

   int k;
   int Mask_radius = filter_width/2;
   int w= (TILE_WIDTH + filter_width - 1);
   
   extern __shared__ float N_ds[];
   
   for (k = 0; k < channels; k++) {
      // First batch loading
      int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
         destY = dest / w,
         destX = dest % w,
         srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius,
         srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius,
         src = (srcY * width + srcX) * channels + k;
         
      if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
         N_ds[destY*w+destX] = I[src];
      else
         N_ds[destY*w+destX] = 0;

      // Second batch loading
      int dest1 = dest + TILE_WIDTH * TILE_WIDTH;
      destY = dest1 / w, 
      destX = dest1 % w,
      srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius,
      srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius,
      src = (srcY * width + srcX) * channels + k;
      
      if (destY < w) {
         if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY*w+destX] = I[src];
         else
            N_ds[destY*w+destX] = 0;
      }
      __syncthreads();

      float accum = 0;
      int y, x;
      for (y = 0; y < filter_width; y++)
         for (x = 0; x < filter_width; x++)
            accum += N_ds[(threadIdx.y + y)*w+(threadIdx.x + x)] * M[y * filter_width + x];
      y = blockIdx.y * TILE_WIDTH + threadIdx.y;
      x = blockIdx.x * TILE_WIDTH + threadIdx.x;
      if (y < height && x < width)
         P[(y * width + x) * channels + k] = accum;
      __syncthreads();
   }
}

int main(int argc, char **argv)
{
	if(argc!=3) {
	cout<<"Program takes two image filenames as parameters"<<endl;
	exit(3);
	}

	float *imgIn, *imgOut;
	int nCols, nRows, channels;
	
	
	// Allocate images and initialize from file
	imgIn = read_image_asfloat(argv[1],&nCols, &nRows, &channels);
	int imgSize = nCols * nRows * channels;
	

	const int blockSize = 256, nStreams = 4;
   const int n = imgSize * blockSize * nStreams;
   const int streamSize = n / nStreams;
   const int streamBytes = streamSize * sizeof(float);
   //const int bytes = n * sizeof(float);
	//imgOut = (float *)calloc(nCols * nRows * channels, sizeof(float));	


  // Allocations on host
  //@TODO@ : 
	cudaHostAlloc((void **) &imgOut, imgSize * sizeof(int), cudaHostAllocDefault);

                                        
	// blur mask
   int filter_width = 7; 
   float maskData[filter_width * filter_width];
   for(int i=0; i<filter_width * filter_width; i++) 
     maskData[i] = 1.0/((float)filter_width * filter_width); 

  
     
	// Allocates device images
	//float *d_imgIn, *d_imgOut, *d_MaskData;

	cudaStream_t streams[nStreams];
	float *d_imgIn[nStreams];
	float *d_imgOut[nStreams];
	float *d_MaskData[nStreams];

	 for(int i=0;i<nStreams;i++)
  {
  		
		cudaStreamCreate(&streams[i]); 
		//@TODO@ : Complete for device allocations
		cudaMalloc(&d_imgIn[i], nCols * nRows * channels * sizeof(float));
		cudaMalloc(&d_imgOut[i], nCols * nRows * channels * sizeof(float));
		cudaMalloc(&d_MaskData[i], filter_width * filter_width * sizeof(float));
	}


	//int dim=32;
	dim3 DimGrid(1 + (nCols-1)/TILE_WIDTH, 1 + (nRows-1)/TILE_WIDTH);
	//int w = TILE_WIDTH - filter_width + 1;
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH);
	
	/*dim3 DimGrid((nCols-1)/dim+ 1, (nRows-1)/dim+1, 1);
	dim3 DimBlock(dim, dim, 1);*/
	size_t nbytes = DimBlock.x*DimBlock.y*sizeof(float);
	for (int i=0; i<n; i+= streamSize) 
	{
	  for(int j=0;j<nStreams;j++)
	 {
	 		//int offset = j * streamSize;
	 		
		   cudaMemcpyAsync(d_imgIn[j], imgIn+i+blockSize*j, streamBytes, cudaMemcpyHostToDevice, streams[j] );
		   
		cudaMemcpyAsync(d_MaskData[j], maskData+i+blockSize*j, filter_width * filter_width * sizeof(float), cudaMemcpyHostToDevice, streams[j] );

	convolution_2D_tiled_kernel<<<DimGrid,DimBlock,nbytes,streams[j]>>>(d_imgIn[j], d_MaskData[j], d_imgOut[j], channels, nCols, nRows, filter_width);
			
		  cudaMemcpyAsync(imgOut+i+blockSize*j, d_imgOut[j], streamBytes, cudaMemcpyDeviceToHost, streams[j]);
	 	}
	 }
	
	cudaDeviceSynchronize();

	// Write gray image to file
	write_image_fromfloat(argv[2], imgOut, nCols, nRows, channels);

	// Free device memory
  for(int i=0;i<nStreams;i++)
	{
		cudaStreamDestroy(streams[i]);
		cudaFree(d_imgIn[i]); 
		cudaFree(d_imgOut[i]); 
		cudaFree(d_MaskData[i]);
	}


	
	// Free host memory
	cudaFreeHost(imgIn); 
	cudaFreeHost(imgOut); 

	
	return 0;
}
