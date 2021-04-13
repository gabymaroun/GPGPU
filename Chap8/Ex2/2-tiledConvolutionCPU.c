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


#define maskCols 25
#define maskRows 25
#define Mask_width  25
#define Mask_radius Mask_width/2
#define TILE_WIDTH 32
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

using namespace std;

//@TODO@ : Write the kernel here

__global__ void convolution_2D_tiled_kernel(float *I, const float* __restrict__ M, float *P, int channels, int width, int height) {

   __shared__ float N_ds[w*w];

   int k;
   
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
      for (y = 0; y < Mask_width; y++)
         for (x = 0; x < Mask_width; x++)
            accum += N_ds[(threadIdx.y + y)*w+(threadIdx.x + x)] * M[y * Mask_width + x];
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
	
	imgOut = (float *)calloc(nCols * nRows * channels, sizeof(float));	


                                        
	// blur mask
  /* int filter_width = 7; */
   float maskData[maskRows * maskCols];
   for(int i=0; i<maskRows * maskCols; i++) 
     maskData[i] = 1.0/((float)maskRows * maskCols); 

  //edge detection
  // int filter_width = 3;
  // float maskData[maskRows * maskCols] = {-1.0,-1.0,-1.0,
  //                                          -1.0, 8.0,-1.0,
  //                                          -1.0,-1.0,-1.0};

  /* // unsharp */
  // int filter_width = 5;
  /*float maskData[maskRows * maskCols] = 	{
  		-1/256.0,-4/256.0,-6/256.0,-4/256.0,-1/256.0,
     -4/256.0,-16/256.0,-24/256.0,-16/256.0,-4/256.0,
     -6/256.0,-24/256.0,476/256.0,-24/256.0,-6/256.0,
     -4/256.0,-16/256.0,-24/256.0,-16/256.0,-4/256.0,
     -1/256.0,-4/256.0,-6/256.0,-4/256.0,-1/256.0};
     */
     
     
	// Allocates device images
	float *d_imgIn, *d_imgOut, *d_MaskData;



	//@TODO@ : Complete for device allocations
	cudaMalloc(&d_imgIn, nCols * nRows * channels * sizeof(float));
	cudaMalloc(&d_imgOut, nCols * nRows * channels * sizeof(float));
	cudaMalloc(&d_MaskData, maskRows * maskCols * sizeof(float));

	// Copy input data
	//@TODO@ : Complete for data copy
	cudaMemcpy(d_imgIn, imgIn, nCols * nRows * channels * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy(d_MaskData, maskData, maskRows * maskCols * sizeof(float), cudaMemcpyHostToDevice );


	// Call the kernel
	//@TODO@ : Compute threads block and grid dimensions
	//@TODO@ : Call the CUDA kernel
	dim3 DimGrid(ceil((float)nCols/TILE_WIDTH), ceil((float)nRows/TILE_WIDTH));
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  // ttype tstart = gettime();
	convolution_2D_tiled_kernel<<<DimGrid,DimBlock>>>(d_imgIn, d_MaskData, d_imgOut, channels, nCols, nRows);
	/*ttype tstop = gettime();
  cout << "Convolution run in "<< (tstop-tstart)*1e-9 << " s."<<endl;	
	*/
	
	// Copy output data
	//@TODO@ : Complete for data copy
	cudaMemcpy(imgOut, d_imgOut, nCols * nRows * channels * sizeof(float), cudaMemcpyDeviceToHost );

	// Write gray image to file
	write_image_fromfloat(argv[2], imgOut, nCols, nRows, channels);

	// Free memory
	//@TODO@ : Free host and device memory
	// Check result
	// check(host_c,n);

	// Free device memory
	cudaFree(d_imgIn); 
	cudaFree(d_imgOut); 	
	cudaFree(d_MaskData); 

	// Free host memory
	free(imgIn); 
	free(imgOut); 

	
	return 0;
}
