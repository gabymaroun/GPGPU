#include <iostream>
#include <cstdlib>
#include <string>

#include "img_utils.hpp"
#define CHANNELS 3

using namespace std;

//@TODO@ : Write the kernel here


__global__ void colorConvert(float * grayImage, float * rgbImage, int width, int height) {

 int x = threadIdx.x+ blockIdx.x* blockDim.x;
 int y = threadIdx.y+ blockIdx.y* blockDim.y;
	//printf("x %d", x);
	//printf("y %d", y);
 if (x < width && y < height) {
  // get 1D coordinate for the grayscale image
  int grayOffset= y * width + x;
	//printf("grayO %d", grayOffset);
  // one can think of the RGB image having
  // CHANNEL times columns than the gray scale image

  int rgbOffset = grayOffset*CHANNELS;
  float r = rgbImage[rgbOffset]; // red value for pixel
  float g = rgbImage[rgbOffset+ 1]; // green value for pixel
  float b = rgbImage[rgbOffset+ 2]; // blue value for pixel
//	printf("r %f", r);
//	printf("g %f", g);
//	printf("b %f", b);
  // perform the rescaling and store it
  // We multiply by floating point constants

  grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
	//printf("grayI %d", grayImage[grayOffset]);
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
	  if(channels!=3){
		cout<<"Input image is not a colored image"<<endl;
		exit(4);
		}
  imgOut = (float *)calloc(nCols*nRows, sizeof(float));

  // Allocates device images
  float *d_imgIn, *d_imgOut;

  //@TODO@ : Complete for device allocations
   cudaMalloc(&d_imgIn, nCols * nRows * channels * sizeof(float));
   cudaMalloc(&d_imgOut, nCols * nRows * sizeof(float));


  // Copy input data
  //@TODO@ : Complete for data copy
   cudaMemcpy(d_imgIn, imgIn, nCols * nRows * channels * sizeof(float), cudaMemcpyHostToDevice );


  // Call the kernel
  //@TODO@ : Compute threads block and grid dimensions
  //@TODO@ : Call the CUDA kernel
  dim3 DimGrid((nRows-1)/32+ 1, (nCols-1)/32+1, 1);
  dim3 DimBlock(32, 32, 1);
  colorConvert<<<DimGrid,DimBlock>>>(d_imgOut, d_imgIn, nCols, nRows);

  // Copy output data
  //@TODO@ : Complete for data copy
  cudaMemcpy(imgOut, d_imgOut, nCols * nRows * sizeof(float), cudaMemcpyDeviceToHost );

  // Write gray image to file
  write_image_fromfloat(argv[2], imgOut, nCols, nRows, 1);
	printf("diO %f",d_imgOut);
	printf("diI %f",d_imgIn);
	printf("iO %f",imgOut);	
	printf("iI %f \n",imgIn);

  // Free memory
  //@TODO@ : Free host and device memory

  // Free device memory
  cudaFree(d_imgIn); 
  cudaFree(d_imgOut); 

  // Free host memory
  free(imgIn); 
  free(imgOut); 

  return 0;
}
