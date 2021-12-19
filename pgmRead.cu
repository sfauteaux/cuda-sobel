#include <stdio.h>
#include <math.h>
#include "pgmRead.h"

/*
pgmInput reads the data from the input .PGM file, extracting the header values and storing them for output later on.
The 2D pixel array is linearized and stored in a 1D array
*/
int * pgmInput(char **header, int *numRows, int *numCols, FILE *in){
    //Initialize storage for the inputs from the file
    char inp[4];
    char buff[100];

    //Store first two header values into 2D header array
    int i = 0;
    while(i<2){
        fgets(buff, sizeof(buff), in);
        strcpy(*(header+i),buff);
        i++;
    }

    //numCols comes first in the file
    fscanf(in, "%s", inp);
    (*numCols) = atoi(inp);

    //numRows is next
    fscanf(in, "%s", inp);
    (*numRows) = atoi(inp);
    
    //save numCols, numRows into header
    char header3[10];
    sprintf(header3, "%d %d",*numCols, *numRows);
    strcpy(*(header+2), header3);


    //maxVal final line of the header
    int maxVal;
    fscanf(in, "%s", inp);
    maxVal = atoi(inp);


    //save maxVal into header
    char header4[5];
    sprintf(header4, "%d",maxVal);
    strcpy(*(header+3), header4);

    const int size = (*numCols) * (*numRows);
    int *input = (int*) malloc(sizeof(int)*size);

    //Read in rest of file into an int array, this will be the 2D int array
    //that we linearize into 1D
    i = 0;
    while(fscanf(in, "%s",inp)!= -1){
        *(input+i) = atoi(inp);
        i++;
    }

    return input;
}

/*
pgmOutput takes the final pixel array and writes to an output file.
*/
int pgmOutput(const char **header, int *pixels, int numRows, int numCols, FILE *out){
    fprintf(out, *header);
    fprintf(out, *(header+1));
    fprintf(out, *(header+2));
    fprintf(out, "\n");
    fprintf(out, *(header+3));
    fprintf(out, "\n");

    int size = numRows*numCols;
    int i;

    //Here, I did my best to mimic the formatting in typical PGM files.
    //It does not look as pretty when viewed as plaintext, but it works and that's what matters.
    for(i=0;i<size;i++){
        if((i>0 && i%12 == 0)){
            fprintf(out, "\n%d  ",pixels[i]);
        }
        else{
            fprintf(out, "%d  ",pixels[i]);
        }
    }

    return 0;
}