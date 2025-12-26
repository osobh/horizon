#!/usr/bin/env python3
"""
VM Operations Script for Kernel Module Testing
"""
import os
import sys
import time
import subprocess

# VM credentials
VM_HOST = "localhost"
VM_PORT = 2222
VM_USER = "osobh"
VM_PASS = "clouddev249"

def run_ssh_batch(commands_file):
    """Run a batch of commands via SSH"""
    print(f"Executing commands from {commands_file}")
    
    # Create SSH command with password via stdin
    ssh_cmd = [
        "ssh",
        "-p", str(VM_PORT),
        "-o", "StrictHostKeyChecking=no",
        "-o", "PreferredAuthentications=password",
        f"{VM_USER}@{VM_HOST}",
        f"bash -s < {commands_file}"
    ]
    
    # Execute with password on stdin
    proc = subprocess.Popen(
        ssh_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send password when prompted
    stdout, stderr = proc.communicate(input=VM_PASS + "\n")
    
    print("Output:", stdout)
    if stderr:
        print("Errors:", stderr)
    
    return proc.returncode == 0

# Create batch command files
print("Creating batch command files...")

# 1. Install dependencies
with open("vm_install_deps.sh", "w") as f:
    f.write("""#!/bin/bash
echo "=== Installing Kernel Development Tools ==="
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) build-essential make gcc
echo "=== Installation Complete ==="
ls -la /lib/modules/$(uname -r)/build
""")

# 2. Build kernel module
with open("vm_build_module.sh", "w") as f:
    f.write("""#!/bin/bash
echo "=== Building Kernel Module ==="
cd ~/gpu_dma_lock
make clean
make
ls -la *.ko
echo "=== Build Complete ==="
""")

# 3. Test kernel module  
with open("vm_test_module.sh", "w") as f:
    f.write("""#!/bin/bash
echo "=== Testing Kernel Module ==="
cd ~/gpu_dma_lock

# Load module
sudo insmod gpu_dma_lock.ko debug=1
lsmod | grep gpu_dma_lock

# Check proc interface
ls -la /proc/swarm/gpu/ || echo "Proc interface not created"

# Run tests
if [ -f tests/test_module.sh ]; then
    ./tests/test_module.sh
else
    echo "Test script not found, running manual tests..."
    cat /proc/swarm/gpu/stats || echo "Stats not readable"
    cat /proc/swarm/gpu/quotas || echo "Quotas not readable"
fi

# Unload module
sudo rmmod gpu_dma_lock
echo "=== Test Complete ==="
""")

print("Batch files created")
print("\nTo use:")
print("1. First copy the archive: scp -P 2222 gpu_dma_lock.tar.gz osobh@localhost:~/")
print("2. Extract in VM: ssh -p 2222 osobh@localhost 'tar -xzf gpu_dma_lock.tar.gz'")
print("3. Run this script to execute operations")