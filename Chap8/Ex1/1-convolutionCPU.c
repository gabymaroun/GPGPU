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
}
*/

#define dim 32
#define maskCols 5
#define maskRows 5

using namespace std;


//@TODO@ : Write the kernel here

__global__ void convolution_1D_basic_kernel(float * I, const float *__restrict__ M,
		float* P, int channels, int width, int height){


	float accum;
	int col = threadIdx.x + blockIdx.x * blockDim.x;   //col index
	int row = threadIdx.y + blockIdx.y * blockDim.y;   //row index
	int maskRowsRadius = maskRows/2;
	int maskColsRadius = maskCols/2;

	int l;
	for (l = 0; l < channels; l++){     //cycle on kernel channels
		
		if(row < height && col < width ){
		
			accum = 0;
			int startRow = row - maskRowsRadius;  //row index shifted by mask radius
			int startCol = col - maskColsRadius;  //col index shifted by mask radius

			for(int i = 0; i < maskRows; i++){ //cycle on mask rows

				for(int j = 0; j < maskCols; j++){ //cycle on mask columns

					int currentRow = startRow + i; // row index to fetch data from input image
					int currentCol = startCol + j; // col index to fetch data from input image

					if(currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width){

							accum += I[(currentRow * width + currentCol )*channels + l] *
										M[i * maskRows + j];
					}
					else accum = 0;
				}

			}
			P[(row* width + col) * channels + l] = accum;
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


                                        
	// blur mask
  /* int filter_width = 7; */
   /*float maskData[maskRows * maskCols];
   for(int i=0; i<maskRows * maskCols; i++) 
     maskData[i] = 1.0/((float)maskRows * maskCols); */

  //edge detection
  // int filter_width = 3;
   /*float maskData[maskRows * maskCols] = {-1.0,-1.0,-1.0,
                                            -1.0, 8.0,-1.0,
                                            -1.0,-1.0,-1.0};
*/

  /* // unsharp */
  // int filter_width = 5;
  float maskData[maskRows * maskCols] = 	{
  		-1/256.0,-4/256.0,-6/256.0,-4/256.0,-1/256.0,
     -4/256.0,-16/256.0,-24/256.0,-16/256.0,-4/256.0,
     -6/256.0,-24/256.0,476/256.0,-24/256.0,-6/256.0,
     -4/256.0,-16/256.0,-24/256.0,-16/256.0,-4/256.0,
     -1/256.0,-4/256.0,-6/256.0,-4/256.0,-1/256.0};
    
     
     
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
	dim3 DimGrid((nCols-1)/dim+ 1, (nRows-1)/dim+1, 1);
	dim3 DimBlock(dim, dim, 1);
	
  // ttype tstart = gettime();
	convolution_1D_basic_kernel<<<DimGrid,DimBlock>>>(d_imgIn, d_MaskData, d_imgOut, channels, nCols, nRows);
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
	//delete(maskData);
	//free(maskData);
	
	return 0;
}
