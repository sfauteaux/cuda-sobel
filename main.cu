/*
Sean Fauteaux
12/7/2021
CSCD 445
Final Project: Sobel Filter

This program uses the Sobel operator to find all the significant hard edges of an image file. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// #include <sys\timeb.h>

#include "cudaDriver.h"
#include "cudaProcess.h"
#include "pgmRead.h"


#define BILLION 1000000000.0

#ifndef M_PI
#define M_PI 3.1415926535897932
#endif

void usage(){
    printf("Usage:\n./project inputFileName\n");
    printf("For the filename, the extension must also be included. \n");
    printf("Example: ./project bloons.pgm\n");
}

double * getGaussKernel(int ksize);
int * gaussBlur(int * input, double * gaussKernel, int ksize, int numRows, int numCols);
void intensityGradient(int * smooth, int numRows, int numCols, int ** xgradient, int ** ygradient, int ** grad);

int main(int argc, char *argv[]){

    if(argc!=2){
        usage();
        return -1;
    }


    FILE *f1 = fopen(argv[1],"r");
    if(f1==NULL){
        printf("No such file found\n");
        return -1;
    }

    char **header = (char**) malloc(sizeof(char*)*4); // initialize 2d array
    header[0] = (char*) malloc(sizeof(char)*3);// 1st header line is just 2 chars + null terminator
    header[1] = (char*) malloc(sizeof(char)*100);// 2nd header line, the comment line
    header[2] = (char*) malloc(sizeof(char)*12);// 3rd header line, the dimensions of the image
    header[3] = (char*) malloc(sizeof(char)*6);// 4th header line, the maximum pixel intensity
    header[4] = (char*) malloc(sizeof(char)*30);// 5th header line (NOT FROM FILE) contains filename

    strcpy(header[4],argv[1]);

    int numRows;
    int numCols;
    int *pixels = pgmInput(header, &numRows, &numCols, f1);

    fclose(f1);

    int *blur, *dx, *dy, *mag;

    int serial = 1;
    int cuda = 1;

    //initialize gaussKernel to be used in both serial and CUDA implementation
    double * gaussKernel = getGaussKernel(2);

    if(serial == 1){
        // struct timeb start, end;
        // ftime(&start);
        clock_t start,end;
        float time;
        start = clock();

        //first step: apply gaussian blur to the image
        blur = gaussBlur(pixels, gaussKernel, 2, numRows, numCols);

        // next: find intensity gradient strength for the x, y directions,
        // save those images in dx, dy, grad
        intensityGradient(blur, numRows, numCols, &dx, &dy, &mag);
        end = clock();

        time = ((float)(end-start) / CLOCKS_PER_SEC) * 1000; // gives time in milliseconds

        // ftime(&end);
        // float elapsed = (float)(1000.0 * (end.time - start.time) + (end.millitm = start.millitm));
        printf("\nCPU runtime in ms = %f\n",time);
    }

    if(cuda==1){
        cudaMain(header, pixels, gaussKernel, 2, numRows, numCols);
    }
    
    const char **hptr = (const char**)header;

    
    // Filenames for all serial outputs
    char cpublur[50];
    char cpudx[50];
    char cpudy[50];
    char cpumag[50];

    strcpy(cpublur,"cpublur_");
    strcpy(cpudx,"cpudx_");
    strcpy(cpudy,"cpudy_");
    strcpy(cpumag,"cpumag_");

    strcat(cpublur,argv[1]);
    strcat(cpudx,argv[1]);
    strcat(cpudy,argv[1]);
    strcat(cpumag,argv[1]);

    FILE *f2 = fopen(cpublur,"w+");
    pgmOutput(hptr, blur, numRows, numCols, f2);
    fclose(f2);

    FILE *f3 = fopen(cpudx,"w+");
    pgmOutput(hptr, dx, numRows, numCols, f3);
    fclose(f3);

    FILE *f4 = fopen(cpudy,"w+");
    pgmOutput(hptr, dy, numRows, numCols, f4);
    fclose(f4);

    FILE *f5 = fopen(cpumag,"w+");
    pgmOutput(hptr, mag, numRows, numCols, f5);
    fclose(f5);

    /* *********************************************
    End of main function, free allocated memory
    ********************************************** */

    int i;
    for(i=0;i<5;i++){
        free(header[i]);
    }

    free(pixels);
    free(blur);
    free(dx);
    free(dy);
    free(mag);
    free(header);

    return 0;
}

/*
Applies a gaussian blur filter to the image for noise reduction.
This is important in helping removing soft edges for future steps. 
*/
int * gaussBlur(int * input, double * gaussKernel, int ksize, int numRows, int numCols){
    int i,j,x,y,sum, k = ksize*2+1;
    //Allocate memory to apply the gaussian blur to the image
    int * gaussImage = (int*) malloc(sizeof(int)*numRows*numCols);
    //For each pixel, we take the sum of all the surrounding pixels
    //multiplied by the gaussian kernel, then store that sum as the new blurred value
    double offset = ksize;
    //ix, jy are used to determine the surrounding pixel locations being used by the filter
    int ix, jy, pid;
    for(i=offset;i<numCols-offset;i++){
        for(j=offset;j<numRows-offset;j++){
            sum = 0;
            for(x=-ksize;x<=ksize;x++){
                for(y=-ksize;y<=ksize;y++){
                    ix = i + x;
                    jy = j + y;
                    pid = (jy*numCols) + ix; 
                    sum += input[pid] * gaussKernel[(x+ksize)*k + (y+ksize)];
                }
            }
            gaussImage[j*numCols + i] = (int)sum;
        }
    }
    return gaussImage;
}

// First, make the (2k+1) x (2k+1) gaussian kernel
// (3x3, 5x5, 7x7, etc)
// Assuming std. deviation is 1
double * getGaussKernel(int ksize){

    int k = ksize*2+1;
    double * gaussKernel = (double *)malloc(sizeof(double) *k*k);

    //r = sqrt(x^2 + y^2)
    double val, sum = 0, r;

    int x, y, i, j;

    //Kernel before normalization
    for(x=-ksize;x<=ksize;x++){
        for(y=-ksize;y<=ksize;y++){
            i = x + ksize;
            j = y + ksize;
            r = sqrt(x*x + y*y);
            //val = (e^(r^2 / 2)) / (2pi)
            val = exp(-(r * r) / 2) / (M_PI * 2);
            gaussKernel[i * k + j] = val;
            sum += val;
        }
    }
        
    //Normalization
    for(i=0;i<(k * k);i++){
        gaussKernel[i] = gaussKernel[i] / sum;
    }
    //Gaussian kernel is now done

    return gaussKernel;
}

void intensityGradient(int * pixels, int numRows, int numCols, int ** dx, int ** dy, int ** gradient){
    int size = numRows*numCols;
    // arrays to hold gradients in the x and y directions of the image
    *dx = (int*)malloc(sizeof(int)*size);
    *dy = (int*)malloc(sizeof(int)*size);
    *gradient = (int*)malloc(sizeof(int)*size);
    int i,j,k;

    // x direction gradient
    for(i=0;i<numRows;i++){
        k = i * numCols; // k is the position in the image we're at
        (*dx)[k] = pixels[k+1] - pixels[k];
        k++;
        // iterate through each pixel in the row, calculating difference
        for(j=1;j+1<numCols;j++){
            (*dx)[k] = pixels[k+1] - pixels[k-1];
            k++;
        }
        (*dx)[k] = pixels[k] - pixels[k-1];
    }

    // y direction gradient, very similar logic to x direciton gradient
    for(j=0;j<numCols;j++){
        k = j;
        (*dy)[k] = pixels[k+numCols] - pixels[k]; 
        k += numCols;
        for(i=1;i+1<numRows;i++){
            (*dy)[k] = pixels[k+numCols] - pixels[k-numCols];
            k+=numCols;
        }
        (*dy)[k] = pixels[k] - pixels[k-numCols];
    }

    // calculate the magnitude of the gradients put together
    // g[k] = sqrt (dx^2 + dy^2)
    k=0;
    double x2,y2;
    for(i=0;i<numCols;i++){
        for(j=0;j<numRows;j++){
            x2 = (double)((*dx)[k] * (*dx)[k]);
            y2 = (double)((*dy)[k] * (*dy)[k]);
            (*gradient)[k] = (int)(sqrt(x2 + y2));
            k++;
        }
    }

    // zero edges of image, 5 pixels in on each side
    // that's how much of an offset we had with a gaussian filter of size 5
    for(i=0;i<numCols;i++){
        for(j=0;j<5;j++){
            // top edge
            (*dx)[j* numCols + i] = 0;
            (*dy)[j* numCols + i] = 0;
            (*gradient)[j* numCols + i] = 0;

            // bottom edge
            (*dx)[((numRows-j)*numCols) + i] = 0;
            (*dy)[((numRows-j)*numCols) + i] = 0;
            (*gradient)[((numRows-j)*numCols) + i] = 0;
        }

    }

    for(j=0;j<numRows;j++){
        for(i=0;i<5;i++){
            // left edge
            (*dx)[j*numCols + i] = 0;
            (*dy)[j*numCols + i] = 0;
            (*gradient)[j*numCols + i] = 0;

            // right edge
            (*dx)[j*numCols + (numCols - i)] = 0;
            (*dy)[j*numCols + (numCols - i)] = 0;
            (*gradient)[j*numCols + (numCols - i)] = 0;
        }
    }

    return;
}