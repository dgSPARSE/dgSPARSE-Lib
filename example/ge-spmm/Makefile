all: spmm.out

PROJ_DIR=../../src/ge-spmm
LIB = $(PROJ_DIR)/libgespmm.a

spmm.out: spmm.cu.o $(LIB)
	nvcc -I$(PROJ_DIR) -lcusparse -o $@ $^

%.o: %.cc
	nvcc -std=c++11 -I/usr/local/cuda/include -I$(PROJ_DIR) -c $< 

clean:
	rm -f spmm.out spmm.cu.o