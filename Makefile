CC = gcc 
TARGET := dgsparse.so
BASE_DIR = $(PWD)
DIRS :=src
OBJ_DIR := objs
EXAMPLE_DIR := example
OBJS = $(wildcard $(OBJ_DIR)/*.o)
RM = -rm -rf
MAKE = make

CFLAGS := -Wall -shared -fPIC
DEBUG = n
ifeq ($(DEBUG), y)
CFLAGS += -g
else
CFLAGS += -O2
endif

INCLUDE = -I$(CUDA_HOME)/include -I../../include/
LOADPATH = -L$(CUDA_HOME)/lib64
LIBRARY = -lcudart

.PHONY: exp code clean

$(TARGET): code
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $(OBJS) $(LOADPATH) $(LIBRARY)
	mv $@ lib/

code:
	mkdir -p $(OBJ_DIR)
	$(MAKE) -C $(DIRS)

exp: code
	$(MAKE) -C $(EXAMPLE_DIR)

clean:
	$(MAKE) -C $(DIRS) clean
	$(MAKE) -C $(EXAMPLE_DIR) clean
	$(RM) -rf $(TARGET) $(OBJ_DIR)