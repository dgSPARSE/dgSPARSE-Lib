all: sddmm.out

PROJ_DIR=../../src/sddmm
LIB = $(PROJ_DIR)/libsddmm.a

sddmm.out: sddmm.cu.o $(LIB)
	nvcc -I$(PROJ_DIR) -lcusparse -o $@ $^

%.o: %.cc
	nvcc -std=c++11 -I/usr/local/cuda/include -I$(PROJ_DIR) -c $< 

clean:
	rm -f sddmm.out sddmm.cu.o