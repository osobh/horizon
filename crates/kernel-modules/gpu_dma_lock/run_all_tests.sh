#!/bin/bash
#
# GPU DMA Lock - Complete Test Suite Runner
# Runs unit tests, integration tests, E2E tests, and generates coverage report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Module directory
MODULE_DIR=$(dirname "$0")
cd "$MODULE_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GPU DMA Lock Module - Complete Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}This script must be run as root (for kernel module testing)${NC}"
    exit 1
fi

# Clean previous build
echo -e "${GREEN}[1/6] Cleaning previous build...${NC}"
make -f Makefile.enhanced clean > /dev/null 2>&1 || true
rm -rf coverage_report/

# Build module with coverage
echo -e "${GREEN}[2/6] Building module with coverage support...${NC}"
make -f Makefile.enhanced COVERAGE=1 DEBUG=1

# Load module
echo -e "${GREEN}[3/6] Loading kernel module...${NC}"
if lsmod | grep -q gpu_dma_lock; then
    rmmod gpu_dma_lock
fi
insmod gpu_dma_lock.ko debug=1

# Run unit tests
echo -e "${GREEN}[4/6] Running unit tests...${NC}"
cd tests
make -C /lib/modules/$(uname -r)/build M=$(pwd) clean > /dev/null 2>&1

# Create test Makefile
cat > Makefile << 'EOF'
obj-m += test_gpu_dma_lock.o
all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
EOF

make > /dev/null 2>&1
insmod test_gpu_dma_lock.ko

# Wait for tests to complete
sleep 3

# Check test results in dmesg
echo -e "${YELLOW}Unit test results:${NC}"
dmesg | grep -A 20 "GPU DMA Lock Test Suite" | tail -20

# Unload test module
rmmod test_gpu_dma_lock

# Run integration tests
echo
echo -e "${GREEN}[5/6] Running integration tests...${NC}"
gcc -o integration_test integration_test.c -lpthread -I../../../common
./integration_test

# Run E2E tests
echo
echo -e "${GREEN}[6/6] Running E2E tests...${NC}"
chmod +x e2e_test.sh
./e2e_test.sh

# Generate coverage report
echo
echo -e "${GREEN}Generating coverage report...${NC}"
cd ..
rmmod gpu_dma_lock
gcov gpu_dma_lock.c
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_report

# Extract coverage percentage
COVERAGE=$(lcov --summary coverage.info 2>&1 | grep lines | awk '{print $2}')

# Print final summary
echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Suite Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Code Coverage: ${GREEN}$COVERAGE${NC}"
echo -e "Coverage Report: ${GREEN}$(pwd)/coverage_report/index.html${NC}"
echo
echo -e "${GREEN}✓ All tests completed successfully!${NC}"

# Cleanup
rm -f tests/Makefile tests/*.o tests/*.ko tests/*.mod.* tests/integration_test