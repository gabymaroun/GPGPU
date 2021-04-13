#include <iostream>
#include <cstdlib>
#include <string>

#include <time.h>
#include "img_utils.hpp"


#define dim 32
#define maskCols 5
#define maskRows 5

#define Mask_width  5
#define Mask_radius Mask_width/2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)


using namespace std;


#define clamp(x) ((x>0.0)?((x<1.0)?x:1.0):0.0)


__global__ void convolutionKernel(float * in, const float *__restrict__ M,
		float* out, int colors, int width, int height){


 __shared__ float N_ds[w][w];
   int k=0;

      int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
      
      int destY = dest / w, destX = dest % w;
      int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
      int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
      int src = (srcY * width + srcX) * colors + k;
     
         if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width){
		for (k = 0; k < colors; k++) {
           		 N_ds[destY][destX] = in[src];
}
}
         else{
		for (k = 0; k < colors; k++) {
           		 N_ds[destY][destX] = 0;
  }    }
      __syncthreads();

      dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
      destY = dest / w, destX = dest % w;
      srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
      srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
      src = (srcY * width + srcX) * colors + k;
      if (destY < w) {
         if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width){
		for (k = 0; k < colors; k++) {
           		 N_ds[destY][destX] = in[src];
	}
}
         else{
		for (k = 0; k < colors; k++) {
           		 N_ds[destY][destX] = 0;
      }
}}
      __syncthreads();

      float pValue = 0;
      int y, x;

      for (k = 0; k < colors; k++) {
     	 for (y = 0; y < maskRows; y++)
         	for (x = 0; x < maskCols; x++)
           	   pValue += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskRows + x];
      y = blockIdx.y * TILE_WIDTH + threadIdx.y;
      x = blockIdx.x * TILE_WIDTH + threadIdx.x;
      if (y < height && x < width)
         out[(y * width + x) * colors + k] = clamp(pValue);
}
      __syncthreads();
   }

int main(int argc, char **argv)
{
	if(argc!=3) {
	cout<<"Program takes k and M as parameters"<<endl;
	exit(3);
	}

	float *imgIn, *imgOut;
	int nCols, nRows, channels;
	
	
	imgIn = read_image_asfloat(argv[1],&nCols, &nRows, &channels);

	cudaHostAlloc((void**) &imgOut, channels*nCols*nRows*sizeof(float), cudaHostAllocDefault);



  float maskData[maskRows *maskCols] = 	{
  		-1/256.0,-4/256.0,-6/256.0,-4/256.0,-1/256.0,
     -4/256.0,-16/256.0,-24/256.0,-16/256.0,-4/256.0,
     -6/256.0,-24/256.0,476/256.0,-24/256.0,-6/256.0,
     -4/256.0,-16/256.0,-24/256.0,-16/256.0,-4/256.0,
     -1/256.0,-4/256.0,-6/256.0,-4/256.0,-1/256.0};



       	float *d_MaskData;


	cudaMalloc(&d_MaskData, maskRows * maskCols * sizeof(float));

	cudaMemcpy(d_MaskData, maskData, maskRows * maskCols * sizeof(float), cudaMemcpyHostToDevice );


	int size = (TILE_WIDTH * TILE_WIDTH)*channels*sizeof (float);
	dim3 DimGrid(ceil(nCols/TILE_WIDTH), ceil(nRows/TILE_WIDTH));
    	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);



int NB_STREAM=4;
cudaStream_t streams[NB_STREAM];
float *d_imgIn[NB_STREAM];
float *d_imgOut[NB_STREAM];

int n = 1000;
int STREAM_SIZE = n * NB_STREAM;
int img_out_start;
int img_out_end;
int img_in_start;
int img_in_end;

for(int i=0;i<NB_STREAM;i++)
{
cudaStreamCreate(&streams[i]); 

cudaMalloc(&d_imgIn[i], nCols * nRows  * channels * sizeof(float));
cudaMalloc(&d_imgOut[i], nCols * nRows * channels * sizeof(float));
}


for (int i=0;i<nRows;i+= STREAM_SIZE){
	for(int j=0;j<NB_STREAM;j++){
		
		if (img_out_start<nRows){
			
			if(img_out_start+n <= nRows)
				img_out_end=img_out_start+n;
			else img_out_end=nRows;

			 
			if(img_out_start - Mask_radius >= 0)
				img_in_start = img_out_start - Mask_radius;
			else img_in_start = 0;
	
			if(img_out_end + Mask_radius <= nRows)
				img_in_end = img_out_end + Mask_radius;
			else img_in_end=nRows;

		cudaMemcpyAsync(d_imgIn[j], imgIn +img_in_start*nCols*channels , (img_in_end - img_in_start)*channels*nCols*sizeof(float), cudaMemcpyHostToDevice, streams[j]);

		convolutionKernel<<<DimGrid,DimBlock,size,streams[j]>>>( d_imgIn[j], d_MaskData, d_imgOut[j],channels, nCols, nRows);
		
		cudaMemcpyAsync(imgOut+img_out_start*nCols*channels, d_imgOut[j] , (img_out_end - img_out_start)*channels*nCols*sizeof(float), cudaMemcpyDeviceToHost, streams[j]);
}
		img_out_start=img_out_start+n+1;

	}

}
cudaDeviceSynchronize();
	
	write_image_fromfloat(argv[2], imgOut, nCols, nRows, channels);

	for(int i=0;i<NB_STREAM;i++)
	{
	  cudaStreamDestroy(streams[i]);
	  cudaFree(d_imgOut[i]); 
	  cudaFree(d_imgIn[i]);
	}
	cudaFree(d_MaskData);
	cudaFreeHost(imgIn);
	cudaFreeHost(imgOut);



	return 0;
}
