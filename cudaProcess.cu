#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cudaProcess.h"

/*
Parallel solution for applying the gaussian blur to the image
*/
__global__ void gaussBlurKernel(int * input, double * gaussKernel, int ksize, int numRows, int numCols, int * output){
    extern __shared__ double gkernel[];

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;

    // thread id relative to the block
    int bid = threadIdx.y * blockDim.x + threadIdx.x;

    // overall thread id relative to the pixel array
    int id = iy*numCols + ix;

    // read gaussian filter kernel into shared memory
    if(bid<25){
        gkernel[bid] = gaussKernel[bid];
    }

    __syncthreads();
    
    // apply the filter by checking all surrounding pixels
    // check for the offset. If within these bounds, thread does nothing
    if(!(ix<2 || ix>numCols-2 || iy<2 || iy>numRows-2)){
        double sum = 0;
        int k = ksize*2+1;

        int x,y, tx, ty, pid, a;
        double b;
        for(x=-ksize;x<=ksize;x++){
            for(y=-ksize;y<=ksize;y++){
                tx = ix + x; 
                ty = iy + y;
                pid = (ty*numCols) + tx;
                a = input[pid]; 
                b = gaussKernel[(x+ksize)*k + (y*ksize)]; 
                sum += a*b;
            }
        }

        // thread work complete, save to output array
        output[id] = sum;
    }
}

/*
Parallel solution for finding the x direction gradient
*/
__global__ void xGradientKernel(int * input, int numRows, int numCols, int * output){
    int size = numRows * numCols;
    // Match thread ID to pixel ID within image
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int id = iy*numCols + ix;

    // ensure thread is within image
    if(ix>0 && ix<numCols-3){
        // Each thread looks at left and right neighbors and calculates the difference
        // between the neighbors to determine the gradients
        output[id] = input[id+1] - input[id-1];
    }
}

/*
Parallel solution for finding the y direction gradient (very similar to x)
*/
__global__ void yGradientKernel(int * input, int numRows, int numCols, int * output){
    int size = numRows * numCols;
    // Match thread ID to pixel ID within image
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int id = iy*numCols + ix;

    // ensure thread is within image
    if(iy>0 && iy<numRows-1){
        // Each thread looks at north and south neighbors and calculates the difference
        // between the neighbors to determine the gradients
        output[id] = input[id+numCols] - input[id-numCols];
    }
}

/*
Parallel solution for funding the magnitude of the gradients put together
*/
__global__ void gradientMagnitudeKernel(int * dx, int * dy, int numRows, int numCols, int * output){
    // output(i,j) = sqrt (dx^2 + dy^2) at (i,j)
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int id = iy*numCols + ix;
    if(ix<numCols && iy<numRows){
        double x2,y2;
        x2 = (double)dx[id];
        x2 *= x2;
        y2 = (double)dy[id];
        y2 *= y2;

        output[id] = (int)(sqrt(x2 + y2));
    }
}