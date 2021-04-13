#include <stdio.h>
#include <cuda.h>
#include <time.h>

#include "matmul_utils.hpp"

int main (int argc, char **argv)
{
  if (argc!=4) {printf("Usage : %s [nb of rows for A] [nb of cols for A] [nb of cols for B]\n", argv[0]); exit(2);}

  srand(time(0));

  int numARows = atoi(argv[1]);
  int numAColumns = atoi(argv[2]);
  int numBColumns  = atoi(argv[3]);
  int numBRows = numAColumns;
  int numCRows =  numARows; 
  int numCColumns = numBColumns ;

  printf("Matrix multiplication dimensions : [%d;%d] = [%d;%d] * [%d;%d]\n", numCRows,numCColumns,numARows,numAColumns, numBRows, numBColumns);

   float * a, * b, * c;

   a = (float *)calloc(numARows*numAColumns, sizeof(float));
   b = (float *)calloc(numBRows*numBColumns, sizeof(float));
   c = (float *)calloc(numCRows*numCColumns, sizeof(float));
 
   init(a,b,numARows,numAColumns,numBRows,numBColumns);
	int na=numARows*numAColumns;
	int nb=numAColumns*numBColumns;
	int nc=numARows*numBColumns;
	float cij;
#pragma acc parallel loop copyout(c[0:nc]), copyin(a[0:na], b[0:nb]) 

		for(int i=0; i<numARows ; i++) {
#pragma acc loop
			for (int j=0; j <  numBColumns; j++) {
				cij =0.0;
#pragma acc loop 
				for (int k=0; k< numAColumns; k++) {
		      	cij+= a[i*numAColumns+k]*b[k* numBColumns +j];
		    }
		    c[i*numAColumns+j] = cij;
		  }
		}
	 

  check (a,b,c, numARows, numAColumns, numBRows, numBColumns);


  free(a);free(b);free(c);
  return 0;

}


