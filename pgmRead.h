#include <stdio.h>
#include <math.h>

int * pgmInput(char **header, int *numRows, int *numCols, FILE *in);
int pgmOutput(const char **header, int *pixels, int numRows, int numCols, FILE *out);