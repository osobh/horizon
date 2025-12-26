# Kernel module makefile for gpu_dma_lock
obj-m := gpu_dma_lock.o
gpu_dma_lock-objs := src/main.o src/allocation.o src/dma.o src/context.o src/proc.o

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules_install
	depmod -a

test: all
	# Load module for testing
	sudo insmod gpu_dma_lock.ko
	# Verify proc interface
	ls -la /proc/swarm/gpu/
	# Run basic tests
	sudo ./tests/e2e_test.sh
	# Unload module
	sudo rmmod gpu_dma_lock

benchmark: all
	# Load module with debug enabled
	sudo insmod gpu_dma_lock.ko debug=1
	# Run performance benchmarks
	cargo test --features bench -- --nocapture
	# Unload module
	sudo rmmod gpu_dma_lock

.PHONY: all clean install test benchmark