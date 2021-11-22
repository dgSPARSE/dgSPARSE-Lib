TARGET = libgespmm.a

OBJ_DIR = ../../objs
OBJS =\
csrspmm_non_transpose.cu.o \
csrspmm_parreduce.cu.o \
csrspmm_rowcaching.cu.o \
csrspmm_seqreduce.cu.o \
gespmm_v2.cu.o \
gespmm_csrcoo_v2.cu.o \
gespmm.cc.o

NVCC = nvcc
NVCC_FLAGS =\
-O3  -lcusparse \
-shared \
-Xcompiler "-fPIC" \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 

CC = g++
CC_FLAGS = -O2 -Wall -std=c++11 -fPIC
INC = -I/usr/local/cuda/include -I../../include/

all: $(TARGET)

# $(TARGET): $(OBJS)
# 	$(NVCC) -shared -o $@ $^

$(TARGET) : $(OBJS)
	ar cr $@ $^
	ranlib $@
	cp $^ $(OBJ_DIR)
	
%.cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

%.cc.o: %.cc
	$(CC) $(CC_FLAGS) $(INC) -c $< -o $@

# install: all
# 	mkdir -p lib
# 	cp $(TARGET) lib/$(TARGET)

clean:
	rm -f *.o $(TARGET)