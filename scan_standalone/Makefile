CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_37 -Xcompiler -Wall -Xcompiler -Wextra -m64

scan: main.cu scan.o Makefile
	nvcc -o scan main.cu scan.o $(NVCC_OPTS)

scan.o: scan.cu
	nvcc -c scan.cu $(NVCC_OPTS)

clean:
	rm -f *.o scan
