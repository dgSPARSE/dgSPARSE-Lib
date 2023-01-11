TARGET = libgespmm.a

OBJ_DIR = ../../objs
SRCS = $(wildcard *.cu) $(wildcard *.cc)
OBJS = $(patsubst %cu, %cu.o, $(SRCS)) $(patsubst %cc, %cc.o, $(SRCS))

NVCC = nvcc
NVCC_FLAGS =\
-O3  -lcusparse -lcudart \
-shared -std=c++11\
-Xcompiler "-fPIC" \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86

CC = g++
CC_FLAGS = -O2 -Wall -std=c++11 -fPIC
INC = -I$(CUDA_HOME)/include -I../../include/


all: $(TARGET)

# $(TARGET): $(OBJS)
# 	$(NVCC) -shared -o $@ $^

$(TARGET) : $(OBJS)
	ar cr $@ $^
	ranlib $@
	if [ ! -d $(OBJ_DIR) ]; then mkdir $(OBJ_DIR); else echo "folder exists"; fi
	cp $(OBJS) $(OBJ_DIR)

%.cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

%.cc.o: %.cc
	$(CC) $(CC_FLAGS) $(INC) -c $< -o $@

# install: all
# 	mkdir -p lib
# 	cp $(TARGET) lib/$(TARGET)

clean:
	rm -f *.o $(TARGET)
