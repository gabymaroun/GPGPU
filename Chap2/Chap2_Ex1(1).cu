#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  printf("%s Starting...\n\n", argv[0]);

  // Get the device count
  int deviceCount = 0;
  //@TODO@ : Complete here

  // getCount function
  cudaGetDeviceCount(&deviceCount);

  // Loop over devices
  for (int dev = 0; dev < deviceCount; ++dev) {
    // Set the current device
    cudaSetDevice(dev);

    // Fill the data structure with device properties
    struct cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    //@TODO@ : Complete here

    // Display some properties
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    //@TODO@ : Complete here
    printf("\nDevice %d: \"%ld\"\n", dev, deviceProp.totalGlobalMem);
    printf("\nDevice %d: \"%d\"\n", dev, deviceProp.multiProcessorCount);
    printf("\nDevice %d: \"%d\"\n", dev, deviceProp.clockRate);
    printf("\nDevice %d: \"%d\"\n", dev, deviceProp.maxThreadsPerBlock);
    printf("\nDevice %d: \"%d\"\n", dev, deviceProp.maxThreadsPerMultiProcessor);

    printf("\nDevice %d: \"%d\"\n", dev, deviceProp.maxGridSize[0]);


    printf("\nDevice %d: \"%d\"\n", dev, deviceProp.maxThreadsDim[0]);




  }
  // finish
  exit(EXIT_SUCCESS);
}
