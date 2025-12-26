#!/bin/bash
# Diagnose kernel module build issues

echo "=== Diagnosing Kernel Module Build ==="
echo

echo "1. Check current directory contents:"
echo "ls -la"
echo

echo "2. Check if .ko file exists:"
echo "find . -name '*.ko' -type f"
echo

echo "3. Check Makefile:"
echo "cat Makefile"
echo

echo "4. Check source files:"
echo "ls -la src/"
echo

echo "5. Check for build errors:"
echo "make clean && make V=1"
echo

echo "6. Check kernel version compatibility:"
echo "uname -r"
echo "ls -la /lib/modules/$(uname -r)/build"
echo

echo "7. Check for missing symbols:"
echo "dmesg | tail -20"