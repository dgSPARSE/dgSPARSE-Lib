NVCC = nvcc
OBJ_DIR = ../../objs
SRCS = $(wildcard *.cu)
OBJS = $(patsubst %cu, %o, $(SRCS))

INCLUDE = -I$(CUDA_HOME)/include/ -I../../include/
NVCC_FLAGS =\
-O3  -lcusparse -lcudart\
-shared -std=c++11\
-Xcompiler "-fPIC" \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86

LIBRARY = -lcudart
LOADPATH = -L$(CUDA_HOME)/lib64
TARGET := libsddmm.a
.PHONY : all clean

all : $(TARGET)

%.o:%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE) -c $< -o $@

$(TARGET): $(OBJS)
	ar cr $@ $^
	ranlib $@
	if [ ! -d $(OBJ_DIR) ]; then mkdir $(OBJ_DIR); else echo "folder exists"; fi
	cp $^ $(OBJ_DIR)

clean:
	rm -f $(OBJS) $(TARGET)
