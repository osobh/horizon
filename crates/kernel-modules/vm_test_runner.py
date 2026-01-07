#!/usr/bin/env python3
import subprocess
import time
import os
import sys

def get_vm_password():
    """Get VM password from environment variable"""
    password = os.environ.get('VM_PASSWORD')
    if not password:
        print("ERROR: VM_PASSWORD environment variable is not set.")
        print("Please set it before running this script:")
        print("  export VM_PASSWORD='your_password'")
        sys.exit(1)
    return password

def run_with_password(cmd, password=None):
    """Run command that expects password input"""
    if password is None:
        password = get_vm_password()
    print(f"Executing: {cmd}")
    
    # Use echo to pipe password to ssh/scp
    full_cmd = f"echo '{password}' | {cmd}"
    
    proc = subprocess.Popen(
        full_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = proc.communicate()
    
    if stdout:
        print("Output:", stdout)
    if stderr and "Warning" not in stderr:
        print("Error:", stderr)
    
    return proc.returncode == 0

def main():
    print("=== Starting Kernel Module Testing in VM ===\n")
    
    # Step 1: Copy files
    print("Step 1: Copying files to VM...")
    
    # Try using sshpass if available
    sshpass_check = subprocess.run("which sshpass", shell=True, capture_output=True)
    
    if sshpass_check.returncode == 0:
        print("Using sshpass for authentication...")
        password = get_vm_password()

        # Copy archive
        cmd1 = f"sshpass -p '{password}' scp -P 2222 -o StrictHostKeyChecking=no gpu_dma_lock.tar.gz osobh@localhost:~/"
        subprocess.run(cmd1, shell=True)

        # Copy test script
        cmd2 = f"sshpass -p '{password}' scp -P 2222 -o StrictHostKeyChecking=no kernel_test_all.sh osobh@localhost:~/"
        subprocess.run(cmd2, shell=True)

        # Run tests
        print("\nStep 2: Running tests in VM...")
        cmd3 = f"sshpass -p '{password}' ssh -p 2222 -o StrictHostKeyChecking=no osobh@localhost 'chmod +x kernel_test_all.sh && ./kernel_test_all.sh'"
        result = subprocess.run(cmd3, shell=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    else:
        print("sshpass not found. Please install it:")
        print("  sudo apt-get install sshpass")
        print("\nOr run these commands manually:")
        print("  scp -P 2222 gpu_dma_lock.tar.gz osobh@localhost:~/")
        print("  scp -P 2222 kernel_test_all.sh osobh@localhost:~/")
        print("  ssh -p 2222 osobh@localhost")
        print("  ./kernel_test_all.sh")

if __name__ == "__main__":
    main()