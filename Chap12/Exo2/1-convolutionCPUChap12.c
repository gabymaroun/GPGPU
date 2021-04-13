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


using namespace std;


//@TODO@ : Write the kernel here

__global__ void convolution_1D_basic_kernel(float * I, const float *__restrict__ M,
		float* P, int channels, int width, int height, int filter_width){


	float accum;
	int col = threadIdx.x + blockIdx.x * blockDim.x;   //col index
	int row = threadIdx.y + blockIdx.y * blockDim.y;   //row index
	int maskRowsRadius = filter_width/2;
	int maskColsRadius = filter_width/2;

	int l;
	for (l = 0; l < channels; l++){     //cycle on kernel channels
		
		if(row < height && col < width ){
		
			accum = 0;
			int startRow = row - maskRowsRadius;  //row index shifted by mask radius
			int startCol = col - maskColsRadius;  //col index shifted by mask radius

			for(int i = 0; i < filter_width; i++){ //cycle on mask rows

				for(int j = 0; j < filter_width; j++){ //cycle on mask columns

					int currentRow = startRow + i; // row index to fetch data from input image
					int currentCol = startCol + j; // col index to fetch data from input image

					if(currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width){

							accum += I[(currentRow * width + currentCol )*channels + l] *
										M[i * filter_width + j];
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
	
	
	int imgSize = nCols * nRows * channels;
	

	const int blockSize = 256, nStreams = 4;
   const int n = imgSize * blockSize * nStreams;
   const int streamSize = n / nStreams;
   const int streamBytes = streamSize * sizeof(float);

	//imgOut = (float *)calloc(nCols * nRows * channels, sizeof(float));	                     
	cudaHostAlloc((void **) &imgOut, imgSize*sizeof(int), cudaHostAllocDefault);
	
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
	 cudaMalloc((void**)&d_imgIn[i], imgSize * sizeof(float));
	 cudaMalloc((void**)&d_imgOut[i], imgSize * sizeof(float));
	 cudaMalloc((void**)&d_MaskData[i], filter_width * filter_width * sizeof(float));
	}


	for (int i=0; i<n; i+= streamSize) 
	{
	  for(int j=0;j<nStreams;j++)
	 {
		// Copy input data
		//@TODO@ : Complete for data copy
		cudaMemcpyAsync(d_imgIn[j], imgIn+i+blockSize*j, imgSize * sizeof(float), cudaMemcpyHostToDevice, streams[j] );
		cudaMemcpyAsync(d_MaskData[j], maskData+i+blockSize*j, filter_width * filter_width * sizeof(float), cudaMemcpyHostToDevice, streams[j] );


		// Call the kernel
		//@TODO@ : Compute threads block and grid dimensions
		//@TODO@ : Call the CUDA kernel
		/*dim3 DimGrid((nCols-1)/dim+ 1, (nRows-1)/dim+1, 1);
		dim3 DimBlock(dim, dim, 1);*/
		
	  // ttype tstart = gettime();
		convolution_1D_basic_kernel<<<streamSize/blockSize,blockSize,0,streams[j]>>>(d_imgIn[j], d_MaskData[j], d_imgOut[j], channels, nCols, nRows, filter_width);
		/*ttype tstop = gettime();
	  cout << "Convolution run in "<< (tstop-tstart)*1e-9 << " s."<<endl;
		*/
		
		// Copy output data
		//@TODO@ : Complete for data copy
		cudaMemcpyAsync(imgOut+i+blockSize*j, d_imgOut[j], imgSize * sizeof(float), cudaMemcpyDeviceToHost, streams[j]);
		}
	}
	  cudaDeviceSynchronize();
	
	// Write gray image to file
	write_image_fromfloat(argv[2], imgOut, nCols, nRows, channels);

	// Free memory
	//@TODO@ : Free host and device memory
	// Check result
	// check(host_c,n);
	for(int i=0;i<nStreams;i++)
	{
		cudaStreamDestroy(streams[i]);
		// Free device memory
		cudaFree(d_imgIn[i]); 
		cudaFree(d_imgOut[i]); 	
		cudaFree(d_MaskData[i]); 
	}
	
	// Free host memory
	cudaFreeHost(imgIn); 
	cudaFreeHost(imgOut); 
	//delete(maskData);
	//free(maskData);
	
	return 0;
}
