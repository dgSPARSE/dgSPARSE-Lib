BASE_PATH := $(PWD)/example
SUB_DIR := ge-spmm \

.PHONY : clean $(SUB_DIR)

all : $(SUB_DIR)
	@echo "compile $(MODULE) done!"

$(SUB_DIR):
	$(MAKE) -C $@

.PHONY : clean

clean:
	$(MAKE) -C $(BASE_PATH)/ge-spmm clean