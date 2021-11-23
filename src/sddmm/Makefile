NVCC = nvcc
OBJ_DIR = ../../objs
SRCS = $(wildcard *.cu)
OBJS = $(patsubst %cu, %o, $(SRCS))

INCLUDE = -I/usr/local/cuda-11.1/include/ -I../../include/
NVCC_FLAGS =\
-O3  -lcusparse \
-shared -std=c++11\
-Xcompiler "-fPIC" \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 

LIBRARY = -lcudart
LOADPATH = -L/usr/local/cuda-11.1/lib64
TARGET := libsddmm.a
.PHONY : all clean

all : $(TARGET)

%.o:%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE) -c $< -o $@

$(TARGET): $(OBJS)
	ar cr $@ $^
	ranlib $@
	cp $^ $(OBJ_DIR)

clean:
	rm -f $(OBJS) $(TARGET)