all: spmm.out

PROJ_DIR=../../src/ge-spmm
LIB = $(PROJ_DIR)/libgespmm.a
SO = ../../lib/dgsparse.so

INC = -I$(CUDA_HOME)/include
LOADPATH = -L$(CUDA_HOME)/lib64

spmm.out: spmm.o $(LIB)
	nvcc -I$(PROJ_DIR) $(INC) $(LOADPATH) -lcusparse -o $@ $^

%.o: %.cu
	nvcc -std=c++11 $(INC) -I$(PROJ_DIR) -c $< 

clean:
	rm -f *.out *.o