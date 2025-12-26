#!/bin/bash
# Execute commands via SSH

echo "Testing SSH connection..."
echo "Please enter password when prompted: clouddev249"
echo

# Test connection
ssh -p 2222 osobh@localhost "echo 'SSH connection successful!'; uname -a; ls -la ~/"