#!/bin/bash

# Simple script to prepare files for VM transfer

echo "=== Preparing Kernel Module for VM Transfer ==="
echo

# Create archive
echo "Creating archive..."
tar -czf gpu_dma_lock.tar.gz gpu_dma_lock/

echo "Archive created: gpu_dma_lock.tar.gz"
echo
echo "To transfer to VM, run:"
echo "  scp -P 2222 gpu_dma_lock.tar.gz osobh@localhost:~/"
echo "  (Password: clouddev249)"
echo
echo "Then in the VM:"
echo "  tar -xzf gpu_dma_lock.tar.gz"
echo "  cd gpu_dma_lock"
echo "  make"
echo "  sudo insmod gpu_dma_lock.ko debug=1"
echo "  ./tests/test_module.sh"
echo "  ./tests/benchmark.sh"
echo
echo "Files included in archive:"
tar -tzf gpu_dma_lock.tar.gz | head -20