MODULE := src
BASE_PATH := $(PWD)/$(MODULE)
SUB_DIR := ge-spmm \
	 	   sddmm \

.PHONY : clean $(SUB_DIR)

all : $(SUB_DIR)
	@echo "compile $(MODULE) done!"

$(SUB_DIR):
	$(MAKE) -C $@

.PHONY : clean

clean:
	$(MAKE) -C $(BASE_PATH)/ge-spmm clean
	$(MAKE) -C $(BASE_PATH)/sddmm clean
