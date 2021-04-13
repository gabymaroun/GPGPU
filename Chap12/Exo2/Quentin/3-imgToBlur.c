#include <iostream>
#include <cstdlib>
#include <string>
#include <cuda.h>
#include "img_utils.hpp"


#define FILTER_RADIUS 2
#define O_TILE_WIDTH 12
#define TILE_WIDTH (O_TILE_WIDTH + 2*FILTER_RADIUS)


using namespace std;



//@TODO@ : Write the kernel here


#define clamp(x) ((x>0.0)?((x<1.0)?x:1.0):0.0)

//--------------------------------------------------------------------------------------------------------------------------
__global__ void convolution(int colors, float * in, const float * __restrict__ mask, float * out, int maskwidth, int w, int h){
   int Col=threadIdx.x + blockDim.x*blockIdx.x;
   int Row=threadIdx.y + blockDim.y*blockIdx.y;

   if (Col < w && Row < h){

	int N_start_col = Col - (maskwidth/2);
	int N_start_row = Row - (maskwidth/2);
	
	//get surrounding box
	for (int c = 0; c <colors; c++){
        float pixVal = 0.0f;


		for (int j = 0; j<maskwidth; ++j){
			for(int k = 0; k < maskwidth; ++k){
				int curRow = N_start_row + j;
				int curCol = N_start_col + k;
				//verifiy we have valid
				if(curRow > -1 && curRow < h && curCol > -1 && curCol < w){
					//printf("OKKK\n");
					pixVal += in[colors * (curRow* w + curCol)+c] * mask[j*maskwidth+k];
				}
			}
		}

	out[colors*(Row * w + Col)+c] = clamp(pixVal);
	}


	}
}




__global__ void convolution_tiled(int colors, float * in, const float * __restrict__ mask, float * out, int maskwidth, int w, int h){

//

   extern __shared__ float tile[];


	int tx= threadIdx.x;
	int ty = threadIdx.y;
	int row_o= blockIdx.y*O_TILE_WIDTH + ty;
	int col_o= blockIdx.x*O_TILE_WIDTH + tx;
	int row_i= row_o-FILTER_RADIUS;
	int col_i= col_o - FILTER_RADIUS;

		//récupération des canaux dans tile[]
		if((row_i>= 0) && (row_i< h) && (col_i>= 0)  && (col_i< w)) {
			for (int c=0;c<colors;c++){
				tile[colors*(ty*TILE_WIDTH+tx)+c] = in[(row_i* w + col_i)*colors+c];
		
			}
		} 
		else{			

			for (int c=0;c<colors;c++){
				tile[(ty*TILE_WIDTH+tx+c)] = 0.0f;
		
			} 
		}
		
    
	

__syncthreads();


	//calcul de l'output

		if(ty < O_TILE_WIDTH && tx< O_TILE_WIDTH){	
		for (int c=0;c<colors;c++){
	float output = 0.0f;
			for(int i = 0; i < maskwidth; i++) {
				for(int j = 0; j < maskwidth; j++) {
					
				output += mask[i*maskwidth+j] * tile[colors*( (i+ty)*TILE_WIDTH+j+tx)+c] ;
				}
			}
//ecriture de l'ouput
	
	if(row_o< h && col_o< w){

		out[colors*(row_o*w + col_o)+c] = clamp(output);
	}

	}
}
__syncthreads();	

	}






int main(int argc, char **argv) {

  if(argc!=3) {cout<<"Program takes two image filenames as parameters"<<endl;exit(3);}
  float *imgIn, *imgOut;
  int nCols, nRows, channels;

  // Allocate images and initialize from file
  imgIn = read_image_asfloat(argv[1],&nCols, &nRows, &channels);

 //imgOut = (float *)calloc(channels*nCols*nRows, sizeof(float));
 cudaHostAlloc((void**) &imgOut, channels*nCols*nRows*sizeof(float), cudaHostAllocDefault);

  // Allocates device images
  float *d_imgIn, *d_imgOut,*d_mask;
  //@TODO@ : Complete for device allocations
  cudaMalloc(&d_imgIn, channels*nCols*nRows*sizeof(float));
  cudaMalloc(&d_imgOut, channels*nCols*nRows*sizeof(float));
  // Copy input data
  //@TODO@ : Complete for data copy
  cudaMemcpy(d_imgIn,imgIn,channels*nCols*nRows*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_imgOut,imgOut,channels*nCols*nRows*sizeof(float), cudaMemcpyHostToDevice);


  // Call the kernel
  //@TODO@ : Compute threads block and grid dimensions



//-------------------------------------------------------------------------------------
//CHOIX DU FILTRE: 
/*
int filter_width = 3;
float mask[9] =
    {-1,-1,-1,-1,8,-1,-1,-1,-1};
*/



int filter_width = 5;
float mask[25] =
    {1/25.0,1/25.0,1/25.0,1/25.0,1/25.0,
	1/25.0,1/25.0,1/25.0,1/25.0,1/25.0,
	1/25.0,1/25.0,1/25.0,1/25.0,1/25.0,
	1/25.0,1/25.0,1/25.0,1/25.0,1/25.0,
	1/25.0,1/25.0,1/25.0,1/25.0,1/25.0};


/*
int filter_width = 5;
  float mask[25] =
    {-1/256.0,-4/256.0,-6/256.0,-4/256.0,-1/256.0,
     -4/256.0,-16/256.0,-24/256.0,-16/256.0,-4/256.0,
     -6/256.0,-24/256.0,476/256.0,-24/256.0,-6/256.0,
     -4/256.0,-16/256.0,-24/256.0,-16/256.0,-4/256.0,
     -1/256.0,-4/256.0,-6/256.0,-4/256.0,-1/256.0};
*/

  cudaMalloc(&d_mask, filter_width*filter_width*sizeof(float));
  cudaMemcpy(d_mask,mask,filter_width*filter_width*sizeof(float), cudaMemcpyHostToDevice);
//-------------------------------------------------------------------------------------


//EXERCICE 1 CONVOLUTION

/*
  dim3 DimGrid((nCols-1)/32 +1,(nRows-1)/32+1,1);
  dim3 DimBlock(32,32,1);
  printf("(%d %d) (%d %d)\n",DimGrid.x,DimGrid.y,DimBlock.x,DimBlock.y);
  convolution<<<DimGrid,DimBlock>>>(channels, d_imgIn, d_mask, d_imgOut, filter_width, nCols, nRows);
 */

//-------------------------------------------------------------------------------------
//EXERCICE 2: CONVOLUTION_TILED
  int BLOCK_WIDTH = O_TILE_WIDTH + (filter_width-1);
  dim3 DimBlock(BLOCK_WIDTH,BLOCK_WIDTH,1);
  dim3 DimGrid( (nCols-1)/O_TILE_WIDTH+1, (nRows-1)/O_TILE_WIDTH+1);
  printf("(%d %d) (%d %d)\n",DimGrid.x,DimGrid.y,DimBlock.x,DimBlock.y);
  int size = (TILE_WIDTH * TILE_WIDTH)*channels*sizeof (float);
  convolution_tiled<<<DimGrid,DimBlock,size>>>(channels, d_imgIn, d_mask, d_imgOut, filter_width, nCols, nRows);

//-------------------------------------------------------------------------------------
  // Copy output data
  //@TODO@ : Complete for data copy
  cudaMemcpy(imgOut,d_imgOut,channels*nCols*nRows*sizeof(float), cudaMemcpyDeviceToHost);
  // Write blur image to file

  write_image_fromfloat(argv[2], imgOut, nCols, nRows, channels);

  // Free memory
  //@TODO@ : Free host and device memory
  cudaFree(d_imgOut); cudaFree(d_imgIn);cudaFree(d_mask);
  cudaFreeHost(imgOut); 
  cudaFreeHost(imgIn);
  return 0;
}
