/*
Sean Fauteaux
12/7/2021
CSCD 445
"cudaDriver.cu" is the "main" file for the CUDA solution.
It performs the necessary memory allocation and CUDA kernel calls.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "cudaDriver.h"
#include "cudaProcess.h"
#include "pgmRead.h"

int cudaMain(char ** header, int * input, double * gaussKernel, int ksize, int numRows, int numCols){
    float tg = 0, tx = 0, ty = 0, tm = 0;
    int * blur = cudaGaussBlur(input, gaussKernel, ksize, numRows, numCols, &tg);
    int * dx = cudaGradientXY(blur, numRows, numCols, 0, &tx);
    int * dy = cudaGradientXY(blur, numRows, numCols, 1, &ty);
    int * magnitude = cudaGradientMagnitude(dx,dy,numRows,numCols,&tm);

    float time = tg+tx+ty+tm;
    printf("CUDA runtime in ms = %f\n",time);

    const char **hptr = (const char**)header;

    // Filenames for all CUDA outputs
    char cudablur[40];
    char cudadx[40];
    char cudady[40];
    char cudamag[40];

    strcpy(cudablur, "cudablur_");
    strcpy(cudadx, "cudadx_");
    strcpy(cudady, "cudady_");
    strcpy(cudamag, "cudamag_");

    strcat(cudablur, header[4]);
    strcat(cudadx, header[4]);
    strcat(cudady, header[4]);
    strcat(cudamag, header[4]);

    // Open file output streams, send to pgmOutput 

    FILE *f1 = fopen(cudablur, "w+");
    FILE *f2 = fopen(cudadx, "w+");
    FILE *f3 = fopen(cudady, "w+");
    FILE *f4 = fopen(cudamag, "w+");

    pgmOutput(hptr, blur, numRows, numCols, f1);
    pgmOutput(hptr, dx, numRows, numCols, f2);
    pgmOutput(hptr, dy, numRows, numCols, f3);
    pgmOutput(hptr, magnitude, numRows, numCols, f4);

    /* *******************************************
    End of work, close file streams and free memory
    ******************************************* */    

    fclose(f1);
    fclose(f2);
    fclose(f3);
    fclose(f4);

    free(blur);
    free(dx);
    free(dy);
    free(magnitude);

    return 0;
}

/*
Preparation for the CUDA implementation of a gaussian blur. 
int * input: the linearized 2D array of pixels
double * gaussKernel: the gaussian filter that will be applied to the pixels
ksize: size of the gaussian kernel
*/
int * cudaGaussBlur(int * input, double * gaussKernel, int ksize, int numRows, int numCols, float * time){
    // Initialize values for the CUDA kernel
    int k = ksize * 2 + 1;
    k *= k;
    int size = numRows * numCols;

    // Allocate enough memory for both the pixel array, as well as the gaussian filter
    int inBytes = sizeof(int)*size;
    int gBytes =  sizeof(double)*k;

    int *gaussBlur = (int*) malloc(inBytes); // host ptr
    int *dptr = 0; // device pointers
    double *gptr = 0;
    int *outptr = 0;

    cudaMalloc((void**)&dptr, inBytes); // allocate memory for kernel
    cudaMalloc((void**)&gptr, gBytes);
    cudaMalloc((void**)&outptr, inBytes);

    cudaMemcpy(dptr, input, inBytes, cudaMemcpyHostToDevice); // fill device arrays with original values
    cudaMemcpy(gptr, gaussKernel, gBytes, cudaMemcpyHostToDevice);
    cudaMemset(outptr, 0, inBytes);

    if(dptr==0 || gptr==0 || gaussBlur==0){
        printf("Could not allocate memory.\n");
        return NULL;
    }

    // use 16x16 block size so that shared memory is utilized most effeciently 
    dim3 grid,block;
    block.x = 16;
    block.y = 16;

    grid.x = ceil( (float)numCols / block.x);
    grid.y = ceil( (float)numRows / block.y);

    // CUDA runtime measured using cudaEvent_t
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start,0);

    // Begin CUDA Gaussian Blur implementation
    gaussBlurKernel<<<grid,block, gBytes>>>(dptr, gptr, ksize,numRows,numCols, outptr);

    cudaEventRecord(end,0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(time,start,end);

    // Check CUDA errors
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) printf("\n%s\n",cudaGetErrorString(error));

    // Copy CUDA results to our array
    cudaMemcpy(gaussBlur,outptr,inBytes,cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dptr);
    cudaFree(gptr);
    cudaFree(outptr);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return gaussBlur;
}

/*
Preparation for the CUDA implementation of x or y direction gradient.
For x gradient, int xy = 0
For y gradient, int xy = 1
*/
int * cudaGradientXY(int * input, int numRows, int numCols, int xy, float * time){
    int size = numRows*numCols;
    int numBytes = sizeof(int)*size;

    int *dxy = (int*)malloc(numBytes); // host pointer
    int *dptr = 0; // device pointers
    int *outptr = 0;

    cudaMalloc((void**)&dptr,numBytes);
    cudaMalloc((void**)&outptr,numBytes);

    cudaMemcpy(dptr,input,numBytes,cudaMemcpyHostToDevice);
    cudaMemset(outptr,0,numBytes);

    if(dptr==0 || outptr==0){
        printf("Could not allocate memory.\n");
        return NULL;
    }

    // try block size 16 x 16
    dim3 grid,block;
    block.x = 16;
    block.y = 16;

    grid.x = ceil( (float)numCols / block.x);
    grid.y = ceil( (float)numRows / block.y);

    // Measure CUDA runtime
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start,0);

    // Call whichever kernel we need
    if(xy==0)xGradientKernel<<<grid,block>>>(dptr,numRows,numCols,outptr);
    else if(xy==1) yGradientKernel<<<grid,block>>>(dptr,numRows,numCols,outptr);
    
    cudaEventRecord(end,0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(time,start,end);

    // Check CUDA errors
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) printf("\n%s\n",cudaGetErrorString(error));

    // Copy results to host
    cudaMemcpy(dxy,outptr,numBytes,cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dptr);
    cudaFree(outptr);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return dxy;
}

/*
Preparation for the CUDA implementation to find the magnitude of the gradients
*/
int * cudaGradientMagnitude(int * dx, int * dy, int numRows, int numCols, float * time){
    int size = numRows*numCols;
    int numBytes = sizeof(int)*size;

    int *magnitude = (int*)malloc(numBytes); // host pointer
    int *dxptr = 0; // device pointers
    int *dyptr = 0;
    int *outptr = 0;

    // CUDA memory allocations, copying host arrays to device global memory
    cudaMalloc((void**)&dxptr,numBytes);
    cudaMalloc((void**)&dyptr,numBytes);
    cudaMalloc((void**)&outptr,numBytes);

    cudaMemcpy(dxptr, dx, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dyptr, dy, numBytes, cudaMemcpyHostToDevice);
    cudaMemset(outptr,0,numBytes);

    if(dxptr == 0 || dyptr == 0 || outptr == 0){
        printf("Could not allocate memory.\n");
        return NULL;
    }

    // try block size 16
    dim3 grid,block;
    block.x = 16;
    block.y = 16;

    grid.x = ceil( (float)numCols / block.x);
    grid.y = ceil( (float)numRows / block.y);

    // Measure CUDA runtime
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start,0);

    // Start kernel    
    gradientMagnitudeKernel<<<grid,block>>>(dxptr,dyptr,numRows,numCols,outptr);

    cudaEventRecord(end,0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(time,start,end);

    // Check CUDA errors
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) printf("\n%s\n",cudaGetErrorString(error));

    // Copy results to host
    cudaMemcpy(magnitude, outptr,numBytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dxptr);
    cudaFree(dyptr);
    cudaFree(outptr);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return magnitude;
}