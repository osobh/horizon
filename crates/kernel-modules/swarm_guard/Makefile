# Kernel module makefile
obj-m := swarm_guard.o
swarm_guard-objs := src/main.o src/namespace.o src/cgroup.o src/syscall.o src/proc.o

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
	sudo insmod swarm_guard.ko
	# Run tests
	sudo dmesg | tail -20
	# Unload module
	sudo rmmod swarm_guard

.PHONY: all clean install test