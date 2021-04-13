#include <iostream>
#include <cstdlib>
#include <string>


#include "img_utils.hpp"


#define BLUR_SIZE 10 // 1st example of blur
using namespace std;

//@TODO@ : Write the kernel here

__global__ void blurKernel(float * image, float * blurImage, int w, int h, int channels) {
	int Col  = blockIdx.x* blockDim.x+ threadIdx.x;
	int Row  = blockIdx.y* blockDim.y+ threadIdx.y;

	if(Col < w && Row < h) {
		int pixels = 0;
		float r = 0;
		float g = 0;
		float b = 0;
		// Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
		for(int blurRow = -BLUR_SIZE; blurRow< BLUR_SIZE+1; ++blurRow) {
		
			for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
			
				int curRow = Row + blurRow; // can be multiplied to change blur 
				int curCol = Col + blurCol;
				
				// Verify we have a valid image pixel
				if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {

					r += image[(curRow * w + curCol) * channels + 0];
					
					if (channels == 3){
						g += image[(curRow * w + curCol) * channels + 1];
						b += image[(curRow * w + curCol) * channels + 2];					
						}
						
					pixels++;// Keep track of number of pixels in the accumulated total
					}
				}
			}
			
		// Write our new pixel value output
		blurImage[(Row * w + Col) * channels] = (float)(r/ (pixels));
		//blurImage[(Row * w + Col) * channels] = (float)(r/ (pixels*2)); // 2nd example of blur 
		//blurImage[(Row * w + Col) * channels] = (float)((r *2)/ (pixels)); // 3rd example of blur

		if(channels ==3){
			blurImage[(Row * w + Col) * channels + 1] = (float)(g/ pixels);
			blurImage[(Row * w + Col) * channels + 2] = (float)(b/ pixels);
		}


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

	// Allocates device images
	float *d_imgIn, *d_imgOut;

	//@TODO@ : Complete for device allocations
	cudaMalloc(&d_imgIn, nCols * nRows * channels * sizeof(float));
	cudaMalloc(&d_imgOut, nCols * nRows * channels * sizeof(float));


	// Copy input data
	//@TODO@ : Complete for data copy
	cudaMemcpy(d_imgIn, imgIn, nCols * nRows * channels * sizeof(float), cudaMemcpyHostToDevice );



	// Call the kernel
	//@TODO@ : Compute threads block and grid dimensions
	//@TODO@ : Call the CUDA kernel
	dim3 DimGrid((nCols-1)/32+ 1, (nRows-1)/32+1, 1);
	dim3 DimBlock(32, 32, 1);

	blurKernel<<<DimGrid,DimBlock>>>(d_imgIn, d_imgOut, nCols, nRows, channels);
	
	
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

	// Free host memory
	free(imgIn); 
	free(imgOut); 

	return 0;
}
