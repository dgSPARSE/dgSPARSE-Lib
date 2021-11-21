CC = gcc 
TARGET := dgsparse.so
SUB_DIR = src
# LIBS := $(SUB_DIR)/ge-spmm/libgespmm.a  # $(SUB_DIR)/sddmm/libsddmm.a
# LDFLAGS = $(LIBS)
RM = -rm -rf
MAKE = make
NVCC = nvcc

CFLAGS := -Wall
DEBUG = y
ifeq ($(DEBUG), y)
CFLAGS += -g
else
CFLAGS += -O2
endif

# $(TARGET): $(LIBS)
# 	$(NVCC) -shared -o $@ $^

DIRS:=src
.PHONY: $(DIRS) clean

$(DIRS):
	$(MAKE) -C $(DIRS);

clean:
	$(MAKE) -C $(DIRS) clean
	$(RM) $(TARGET)