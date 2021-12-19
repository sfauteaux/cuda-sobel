#ifndef cudaProcess
#define cudaProcess

__global__ void gaussBlurKernel(int * input, double * gaussKernel, int ksize, int numRows, int numCols, int * output);
__global__ void xGradientKernel(int * input, int numRows, int numCols, int * output);
__global__ void yGradientKernel(int * input, int numRows, int numCols, int * output);
__global__ void gradientMagnitudeKernel(int * dx, int * dy, int numRows, int numCols, int * output);

#endif