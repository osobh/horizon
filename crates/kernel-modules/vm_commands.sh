#!/bin/bash
# Commands to run for kernel testing

echo "=== VM Kernel Testing Commands ==="
echo
echo "Please run these commands:"
echo
echo "1. First terminal - Copy files:"
echo "   scp -P 2222 gpu_dma_lock.tar.gz osobh@localhost:~/"
echo "   scp -P 2222 kernel_test_all.sh osobh@localhost:~/"
echo "   Password: clouddev249"
echo
echo "2. Second terminal - Connect and test:"
echo "   ssh -p 2222 osobh@localhost"
echo "   Password: clouddev249"
echo
echo "3. Inside VM:"
echo "   chmod +x kernel_test_all.sh"
echo "   ./kernel_test_all.sh"
echo
echo "The test script will:"
echo "- Install kernel headers"
echo "- Build the kernel module"  
echo "- Load and test the module"
echo "- Run performance benchmarks"
echo "- Generate results"
echo
echo "Alternatively, run step by step:"
echo "   tar -xzf gpu_dma_lock.tar.gz"
echo "   cd gpu_dma_lock"
echo "   sudo apt-get update && sudo apt-get install -y linux-headers-\$(uname -r) build-essential"
echo "   make"
echo "   sudo insmod gpu_dma_lock.ko debug=1"
echo "   ./tests/test_module.sh"
echo "   sudo rmmod gpu_dma_lock"