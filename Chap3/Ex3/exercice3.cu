#include <iostream>
#include <cstdlib>
#include <string>



#include "img_utils.hpp"

#define BLUR_SIZE 25

using namespace std;

__global__ void blurKernel(float* d_imgIn, float * d_imgOut, int h, int w, int channels) {

int col = blockIdx.x*blockDim.x + threadIdx.x;
int row = blockIdx.y*blockDim.y + threadIdx.y;

if((row < h) && (col < w)){

if (channels == 3) {
	float pixVal=0;
	float pixels=0;
	float r=0;
	float g=0;
	float b=0;
	
	
	int rgb = (row * w + col) * 3;


	// Get the average of the surrounding 3xBLUR_SIZE x 3xBLUR_SIZE box : -1 0 1 
	for(int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
		for(int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
			int curRow = row + blurRow;
			int curCol = col + blurCol;
			// Verify we have a valid image pixel
			if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
				r += d_imgIn[(curRow * w + curCol) * 3 + 0];
				g += d_imgIn[(curRow * w + curCol) * 3 + 1];
				b += d_imgIn[(curRow * w + curCol) * 3 + 2];
			
				pixels++; // Keep track of number of pixels in the accumulated total

			}
		}
	}

			// Write our new pixel value out

			d_imgOut[rgb + 0] = (r/(pixels));
			d_imgOut[rgb+ 1] = (g/(pixels));
			d_imgOut[rgb+ 2] = (b/(pixels));
}//for greyscale 
if (channels == 1) {
	float pixVal=0;
	float pixels=0;

	//float pixVal_r=0;
	//float pixVal_g=0;
	//float pixVal_b=0;
	
	//int rgbOffset = (row * width + col) * channels;
	int grayOffset = (row * w + col) * channels;


	// Get the average of the surrounding 3xBLUR_SIZE x 3xBLUR_SIZE box : -1 0 1 
	for(int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
		for(int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {

			int curRow = row + blurRow;
			int curCol = col + blurCol;
			int curOffset = curRow * w + curCol;

			// Verify we have a valid image pixel
			if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
				
				pixVal += d_imgIn[curOffset];
				
				pixels++; // Keep track of number of pixels in the accumulated total

			}
		}
	}

// Write our new pixel value out

d_imgOut[grayOffset] = (float)(pixVal/ pixels);
}

	
	}
}


int main(int argc, char **argv)
{
  if(argc!=3) {cout<<"Program takes two image filenames as parameters"<<endl;exit(3);}

  float *imgIn, *imgOut;
  int nCols, nRows, channels;

  //Allocate images and initialize from file
  imgIn = read_image_asfloat(argv[1],&nCols, &nRows, &channels);
  if(channels!=3){cout<<"Input image is not a colored image"<<endl;exit(4);}
  imgOut = (float *)calloc(nCols*nRows*channels, sizeof(float));

  // Allocates device images
  float *d_imgIn, *d_imgOut;
  //@TODO@ : Complete for device allocations
  cudaMalloc(&d_imgIn, nCols * nRows *  channels * sizeof(float));
  cudaMalloc(&d_imgOut, nCols * nRows*channels *  sizeof(float));
  

  // Copy input data
  //@TODO@ : Complete for data copy
   cudaMemcpy(d_imgIn,imgIn, nCols * nRows * channels* sizeof(float), cudaMemcpyHostToDevice);
   //cudaMemcpy(d_imgOut,imgOut, nCols * nRows *  sizeof(float), cudaMemcpyHostToDevice);

  // Call the kernel
  //@TODO@ : Compute threads block and grid dimensions
   dim3 DimGrid((nRows-1)/32+1 , (nCols-1)/32+1,1);
   dim3 DimBlock (32,32,1);

  //@TODO@ : Call the CUDA kernel
   blurKernel<<< DimGrid, DimBlock >>>(d_imgIn,d_imgOut,nCols,nRows,channels);

  // Copy output data
  //@TODO@ : Complete for data copy
   //cudaMemcpy(imgIn,d_imgIn, nCols * nRows * channels*  sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(imgOut,d_imgOut, nCols * nRows *channels *sizeof(float), cudaMemcpyDeviceToHost);

  // Write gray image to file
   write_image_fromfloat(argv[2], imgOut, nCols, nRows,channels);

  // Free memory
  //@TODO@ : Free host and device memory
  cudaFree(d_imgOut);  cudaFree(d_imgIn);
  // Free host memory
  free(imgOut); free(imgIn); 


  return 0;
}
