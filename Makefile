CC = gcc 
TARGET := dgsparse.so
BASE_DIR = $(PWD)
DIRS :=src
OBJ_DIR := objs
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

INCLUDE = -I/usr/local/cuda/include/ -I../../include/
LOADPATH = -L/usr/local/cuda/lib64
LIBRARY = -lcudart

.PHONY: $(DIRS) clean

$(TARGET): $(DIRS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LOADPATH) $(LIBRARY)
	mv $@ lib/

$(DIRS):
	mkdir -p $(OBJ_DIR)
	$(MAKE) -C $(DIRS)

clean:
	$(MAKE) -C $(DIRS) clean
	$(RM) -rf $(TARGET) $(OBJ_DIR)