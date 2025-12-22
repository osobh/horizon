# GPU DMA Lock Kernel Module Makefile

SHELL := /bin/bash

# Choose implementation
ifeq ($(REAL_GPU),1)
obj-m += gpu_dma_lock_real.o
MODULE_FILE := gpu_dma_lock_real.ko
else
obj-m += gpu_dma_lock.o
MODULE_FILE := gpu_dma_lock.ko
endif

# Build flags
ccflags-y := -I$(src)/../common -Wall -Wextra
ccflags-$(CONFIG_DEBUG) += -DDEBUG -g
ccflags-$(CONFIG_COVERAGE) += -fprofile-arcs -ftest-coverage

# Kernel directory
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# Module info
MODULE_NAME := gpu_dma_lock
MODULE_VERSION := 2.0

# Default target
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

# Clean build artifacts
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f *.gcno *.gcda *.gcov

# Install module
install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a

# Load module
load:
	sudo insmod $(MODULE_FILE) debug=1 gpudirect_enable=1

# Unload module
unload:
	-sudo rmmod $(MODULE_NAME)

# Test targets
test: all
	@echo "Running GPU DMA Lock tests..."
	cd tests && sudo ./run_tests.sh

# Unit tests with coverage
test-unit: CCFLAGS += -DUNIT_TEST
test-unit: clean
	$(MAKE) COVERAGE=1 all
	cd tests && sudo insmod test_gpu_dma_lock.ko
	sleep 2
	cd tests && sudo rmmod test_gpu_dma_lock
	lcov --capture --directory . --output-file coverage.info
	lcov --remove coverage.info '/usr/*' --output-file coverage.info
	genhtml coverage.info --output-directory coverage_report

# Integration tests
test-integration: all load
	cd tests && sudo ./integration_test
	$(MAKE) unload

# E2E tests
test-e2e: all load
	cd tests && sudo ./e2e_test.sh
	$(MAKE) unload

# Performance tests
test-perf: all load
	cd tests && sudo ./perf_test
	$(MAKE) unload

# Run all tests
test-all: test-unit test-integration test-e2e test-perf

# Debug helpers
check-proc: load
	@echo "=== GPU Module Stats ==="
	cat /proc/swarm/gpu/stats
	@echo "=== GPU Agents ==="
	cat /proc/swarm/gpu/agents

# Development helpers
dev-reload: unload all load check-proc

# Build with real GPU support
real:
	$(MAKE) REAL_GPU=1 all

# Build with mock support (default)
mock:
	$(MAKE) REAL_GPU=0 all

# Coverage report
coverage:
	$(MAKE) COVERAGE=1 test-all
	@echo "Coverage report generated in coverage_report/"

.PHONY: all clean install load unload test test-unit test-integration test-e2e test-perf test-all check-proc dev-reload real mock coverage