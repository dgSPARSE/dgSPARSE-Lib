all: sddmm.out

PROJ_DIR=../../src/sddmm
LIB = $(PROJ_DIR)/libsddmm.a

INC = -I$(CUDA_HOME)/include
LOADPATH = -L$(CUDA_HOME)/lib64

sddmm.out: sddmm.o $(LIB)
	nvcc -I$(PROJ_DIR) $(INC) $(LOADPATH) -lcusparse -o $@ $^

%.o: %.cu
	nvcc -std=c++11 $(INC) -I$(PROJ_DIR) -c $< 

clean:
	rm -f sddmm.out sddmm.o