# Kernel module makefile for tier_watch
obj-m := tier_watch.o
tier_watch-objs := src/main.o src/tiers.o src/migration.o src/fault.o src/proc.o

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
	sudo insmod tier_watch.ko
	# Verify proc interface
	ls -la /proc/swarm/tiers/
	# Run basic tests
	sudo ./tests/e2e_test.sh
	# Unload module
	sudo rmmod tier_watch

benchmark: all
	# Load module with debug enabled
	sudo insmod tier_watch.ko debug=1
	# Run benchmarks
	cargo test --features bench -- --nocapture
	# Unload module
	sudo rmmod tier_watch

.PHONY: all clean install test benchmark