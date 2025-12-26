#!/usr/bin/env python3
"""
Automated Kernel Module Testing in VM
"""
import os
import subprocess
import time

VM_PORT = 2222
VM_USER = "osobh"
VM_HOST = "localhost"

def run_command(cmd, desc):
    """Run a command and print results"""
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("Output:")
        print(result.stdout)
    
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    print("=== StratoSwarm Kernel Module Testing ===")
    print("This script will guide you through the testing process")
    print()
    
    # Step 1: Copy archive
    print("Step 1: Copy kernel module archive to VM")
    print("Please run this command in another terminal:")
    print(f"  scp -P {VM_PORT} gpu_dma_lock.tar.gz {VM_USER}@{VM_HOST}:~/")
    print("  Password: clouddev249")
    input("\nPress Enter when the file has been copied...")
    
    # Step 2: Extract archive
    print("\nStep 2: Extract archive in VM")
    print("Please run these commands in the VM:")
    print(f"  ssh -p {VM_PORT} {VM_USER}@{VM_HOST}")
    print("  Password: clouddev249")
    print("  tar -xzf gpu_dma_lock.tar.gz")
    print("  cd gpu_dma_lock")
    input("\nPress Enter when extraction is complete...")
    
    # Step 3: Install dependencies
    print("\nStep 3: Install kernel development tools")
    print("In the VM, run:")
    print("  sudo apt-get update")
    print("  sudo apt-get install -y linux-headers-$(uname -r) build-essential make gcc")
    input("\nPress Enter when installation is complete...")
    
    # Step 4: Build module
    print("\nStep 4: Build kernel module")
    print("In the VM, run:")
    print("  make clean")
    print("  make")
    print("  ls -la *.ko")
    input("\nPress Enter when build is complete...")
    
    # Step 5: Test module
    print("\nStep 5: Test kernel module")
    print("In the VM, run these commands:")
    print()
    print("# Load module")
    print("sudo insmod gpu_dma_lock.ko debug=1")
    print()
    print("# Verify loaded")
    print("lsmod | grep gpu_dma_lock")
    print("sudo dmesg | tail -20")
    print()
    print("# Check proc interface")
    print("ls -la /proc/swarm/gpu/")
    print()
    print("# Run tests")
    print("./tests/test_module.sh")
    print("./tests/benchmark.sh")
    print()
    print("# Unload module")
    print("sudo rmmod gpu_dma_lock")
    
    input("\nPress Enter when testing is complete...")
    
    print("\n=== Testing Complete ===")
    print("Please share the test results!")

if __name__ == "__main__":
    main()