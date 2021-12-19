#ifndef cudaDriver
#define cudaDriver

#include <stdio.h>
#include <math.h>

int cudaMain(char ** header, int * input, double * gaussKernel, int ksize, int numRows, int numCols);
int * cudaGaussBlur(int * input, double * gaussKernel, int ksize, int numRows, int numCols, float * time);
int * cudaGradientXY(int * input, int numRows, int numCols, int xy, float * time);
int * cudaGradientMagnitude(int * dx, int * dy, int numRows, int numCols, float * time);

#endif