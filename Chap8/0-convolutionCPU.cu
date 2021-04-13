#include <iostream>
#include <cstdlib>
#include <string>
#include <time.h>

#include "img_utils.hpp"
typedef unsigned long long ttype;
ttype gettime(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (((ttype)ts.tv_sec*1e9) + ts.tv_nsec);
}


using namespace std;

#define clamp(x) ((x>0.0)?((x<1.0)?x:1.0):0.0)
void convolution(float *out, float *in, int width, int height, int colors,
                 int filter_width, const float* const mask) {
  int i,j;
  for(int col=0; col<width; col++) {
    for(int row=0; row<height; row++) {
      for(int c=0;c<colors;c++) {
        float pxVal = 0.0f;
        // Get the average of the surrounding filter_radius x filter_radius box
        for(i=0; i<filter_width; i++) {
          for(j=0; j<filter_width; j++) {
            int currow = row - (filter_width-1)/2 + i;
            int curcol = col - (filter_width-1)/2 + j;
            // Verify we have a valid image pixel
            if (currow > -1 && currow < height && curcol > -1 && curcol < width) {
              pxVal += in[colors*(currow*width+curcol)+c]*mask[i*filter_width+j];
            }
          }
        }
        out[colors*(row*width+col)+c] = clamp(pxVal);
      }
    }
  }
}
int main(int argc, char **argv)
{
  if(argc!=3) {cout<<"Program takes two image filenames as parameters"<<endl;exit(3);}
  float *imgIn, *imgOut;
  int nCols, nRows, channels;

  //testing
  /* int filter_width=1; */
  /* float mask[1] = {1.0,}; */
  /* int filter_width = 3; */
  /* float mask[filter_width*filter_width] = {0.0,0.0,0.0, */
  /*                                                    0.0,1.0,0.0, */
  /*                                                    0.0,0.0,0.0}; */

  // blur mask
   int filter_width = 25; 
   float mask[filter_width*filter_width]; 
   for(int i=0; i<filter_width*filter_width; i++) 
     mask[i] = 1.0/((float)filter_width*filter_width); 

  //edge detection
  // int filter_width = 3;
  // float mask[filter_width*filter_width] = {-1.0,-1.0,-1.0,
  //                                          -1.0, 8.0,-1.0,
  //                                          -1.0,-1.0,-1.0};

  /* // unsharp */
  /*int filter_width = 5;
  float mask[filter_width*filter_width] =
    {-1/256.0,-4/256.0,-6/256.0,-4/256.0,-1/256.0,
     -4/256.0,-16/256.0,-24/256.0,-16/256.0,-4/256.0,
     -6/256.0,-24/256.0,476/256.0,-24/256.0,-6/256.0,
     -4/256.0,-13/256.0,-24/256.0,-16/256.0,-4/256.0,
     -1/256.0,-4/256.0,-6/256.0,-4/256.0,-1/256.0};*/

  // Allocate images and initialize from file
  imgIn = read_image_asfloat(argv[1],&nCols, &nRows, &channels);
  int imgSize = nCols*nRows*channels;
  imgOut = (float *)calloc(imgSize, sizeof(float));

  // Call the kernel
  ttype tstart = gettime();
  convolution(imgOut, imgIn, nCols, nRows, channels, filter_width, mask);
  ttype tstop = gettime();
  cout << "Convolution run in "<< (tstop-tstart)*1e-9 << " s."<<endl;
  // Write gray image to file
  write_image_fromfloat(argv[2], imgOut, nCols, nRows, channels);

  // Free memory
  free(imgIn); free(imgOut);
  return 0;
}
          
