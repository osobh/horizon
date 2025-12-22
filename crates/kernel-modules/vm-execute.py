#!/usr/bin/env python3
import subprocess
import sys
import time

def execute_ssh_command(command, password="clouddev249"):
    """Execute command via SSH with password authentication"""
    ssh_cmd = f"sshpass -p '{password}' ssh -p 2222 -o StrictHostKeyChecking=no osobh@localhost {command}"
    
    try:
        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
        print(f"Command: {command}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Exception: {e}")
        return False

def copy_file_to_vm(local_file, remote_path="~/", password="clouddev249"):
    """Copy file to VM via SCP with password"""
    scp_cmd = f"sshpass -p '{password}' scp -P 2222 -o StrictHostKeyChecking=no {local_file} osobh@localhost:{remote_path}"
    
    try:
        result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Copied {local_file} to VM")
            return True
        else:
            print(f"✗ Failed to copy {local_file}: {result.stderr}")
            return False
    except Exception as e:
        print(f"Exception copying file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
        execute_ssh_command(command)
    else:
        print("Usage: ./vm-execute.py <command>")