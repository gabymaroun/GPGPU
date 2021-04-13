#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include "img_utils.hpp"

using namespace cv;
using namespace std;

float * read_image_asfloat(char *imgName,int *nC, int *nR, int *colors) {
  Mat I=imread(imgName, IMREAD_UNCHANGED);
  CV_Assert(I.depth() == CV_8U);
  int channels = I.channels();
  int nRows = I.rows;
  int nCols = I.cols * channels;
  *nC=I.cols;
  *nR=I.rows;
  *colors=I.channels();
  float *img;
  cudaHostAlloc((void **) &img, (*nR)*(*nC)*(*colors) * sizeof(int), cudaHostAllocDefault);
  
  if (I.isContinuous())
  {
    nCols *= nRows;
    nRows = 1;
  }
  int i,j;
  uchar* p;
  for( i = 0; i < nRows; ++i)
  {
    p = I.ptr<uchar>(i);
    for ( j = 0; j < nCols; ++j)
    {
      img[i*nCols+j]=((float)p[j])/255.0;
    }
  }
  imwrite(imgName, I);
  cout << "Read image of size "<<*nC<<"x"<<*nR<<" "<<*colors<<" channels"<<endl;
  return img;
}

void write_image_fromfloat(char *filename,float *img, int nC, int nR, int colors) {
  Mat I;
  if(colors==1) {
    I = Mat(nR,nC, CV_8UC1, Scalar(0));
  } else if(colors==3) {
    I = Mat(nR,nC, CV_8UC3, Scalar(0,0,0));
  } else {cout<<"Wrong number of channels"<<endl; exit(5);}
  int channels = I.channels();
  int nRows = I.rows;
  int nCols = I.cols * channels;
  if (I.isContinuous())
  {
    nCols *= nRows;
    nRows = 1;
  }
  int i,j;
  uchar* p;
  for( i = 0; i < nRows; ++i)
  {
    p = I.ptr<uchar>(i);
    for ( j = 0; j < nCols; ++j)
    {
      p[j] = (unsigned char)(img[i*nCols+j]*255);
    }
  }
  cout<<"Write image "<<I.cols<<"x"<<I.rows<<" "<<I.channels()<<" colors into "<< filename<<endl;
  imwrite(filename, I);
}
