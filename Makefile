project: main.o pgmRead.o cudaDriver.o cudaProcess.o
	nvcc -arch=sm_37 -g -G -o project main.o pgmRead.o cudaDriver.o cudaProcess.o -lineinfo -I.

main.o: main.cu
	nvcc -arch=sm_37 -c main.cu

pgmRead.o: pgmRead.cu
	nvcc -arch=sm_37 -c pgmRead.cu

cudaDriver.o: cudaDriver.cu
	nvcc -arch=sm_37 -c cudaDriver.cu

cudaProcess.o: cudaProcess.cu
	nvcc -arch=sm_37 -c cudaProcess.cu

clean:
	rm -r *.o cannyEdge