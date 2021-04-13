#include <iostream>
#include <cstdlib>
#include <string>

#include "img_utils.hpp"


using namespace std;
#define size 32
//#define channels 3

//@TODO@ : Write the kernel here


__global__ void colorConvert(float * grayImage, float * inImage, int width, int height, int channels) {
 
 __shared__ float d_imgIn[size*size*3];
 int x =  blockIdx.x* blockDim.x;
 int y =  blockIdx.y* blockDim.y;
 int col = x + threadIdx.x;
 int row = y + threadIdx.y;
 
 
 if ((col < width) && (row < height)) {
 
 	  d_imgIn[threadIdx.y * channels * size + threadIdx.x] = inImage[(y*width+x)*channels + threadIdx.y * channels * width + threadIdx.x];
 	  d_imgIn[threadIdx.y * channels * size + threadIdx.x + 1*size] = inImage[(y*width+x)*channels + 1*size + threadIdx.y * channels * width + threadIdx.x];
 	  d_imgIn[threadIdx.y * channels * size + threadIdx.x + 2*size] = inImage[(y*width+x)*channels + 2*size + threadIdx.y * channels * width + threadIdx.x];
 	  

	  // one can think of the RGB image having
	  // CHANNEL times columns than the gray scale image 	  
  	  int rgbOffset = (threadIdx.y*size+threadIdx.x)*channels;
	  // get 1D coordinate for the grayscale image
	  int grayOffset= row * width + col;
		//printf("grayO %d", grayOffset);


	  float r = d_imgIn[rgbOffset]; // red value for pixel
	  float g = d_imgIn[rgbOffset+ 1]; // green value for pixel
	  float b = d_imgIn[rgbOffset+ 2]; // blue value for pixel

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
  int nCols, nRows, c;

  // Allocate images and initialize from file
  imgIn = read_image_asfloat(argv[1],&nCols, &nRows, &c);
	/*  if(channels!=3){
		cout<<"Input image is not a colored image"<<endl;
		exit(4);
		}*/
  imgOut = (float *)calloc(nCols*nRows, sizeof(float));

  // Allocates device images
  float *d_imgIn, *d_imgOut;

  //@TODO@ : Complete for device allocations
   cudaMalloc(&d_imgIn, nCols * nRows * c * sizeof(float));
   cudaMalloc(&d_imgOut, nCols * nRows * sizeof(float));


  // Copy input data
  //@TODO@ : Complete for data copy
   cudaMemcpy(d_imgIn, imgIn, nCols * nRows * c * sizeof(float), cudaMemcpyHostToDevice );


  // Call the kernel
  //@TODO@ : Compute threads block and grid dimensions
  //@TODO@ : Call the CUDA kernel
  dim3 DimGrid((nCols-1)/size+ 1, (nRows-1)/size+1, 1);
  dim3 DimBlock(size, size, 1);
  colorConvert<<<DimGrid,DimBlock>>>(d_imgOut, d_imgIn, nCols, nRows, c);

  // Copy output data
  //@TODO@ : Complete for data copy
  cudaMemcpy(imgOut, d_imgOut, nCols * nRows * sizeof(float), cudaMemcpyDeviceToHost );

  // Write gray image to file
  write_image_fromfloat(argv[2], imgOut, nCols, nRows, 1);


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
