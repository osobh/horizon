#!/usr/bin/env python3
"""
Fix incorrectly converted array indexing patterns
"""

import os
import re
import sys
from pathlib import Path

def fix_array_indexing(content):
    """Fix array indexing that was incorrectly converted"""
    
    # Pattern 1: sdata.get(0).ok_or(...)?; -> sdata[0]
    content = re.sub(
        r'(\w+)\.get\((\d+)\)\.ok_or\([^)]+\)\?\[(\d+)\]',
        r'\1[\2][\3]',
        content
    )
    
    # Pattern 2: array.get(index).ok_or(...)? -> array[index] in simple cases
    content = re.sub(
        r'(\w+)\.get\((\d+)\)\.ok_or\([^)]+\)\?([;\s])',
        r'\1[\2]\3',
        content
    )
    
    # Pattern 3: Fix CUDA array declarations
    content = re.sub(
        r'float (\w+)\.get\((\d+)\)\.ok_or\([^)]+\)\?\[(\d+)\];',
        r'float \1[\2][\3];',
        content
    )
    
    content = re.sub(
        r'float (\w+)\.get\((\d+)\)\.ok_or\([^)]+\)\?;',
        r'float \1[\2];',
        content
    )
    
    content = re.sub(
        r'double (\w+)\.get\((\d+)\)\.ok_or\([^)]+\)\?;',
        r'double \1[\2];',
        content
    )
    
    # Pattern 4: __shared__ memory declarations
    content = re.sub(
        r'__shared__ float (\w+)\.get\((\d+)\)\.ok_or\([^)]+\)\?\[(\d+)\];',
        r'__shared__ float \1[\2][\3];',
        content
    )
    
    content = re.sub(
        r'__shared__ float (\w+)\.get\((\d+)\)\.ok_or\([^)]+\)\?;',
        r'__shared__ float \1[\2];',
        content
    )
    
    # Pattern 5: Function calls with array indexing
    content = re.sub(
        r'atomicAdd\(&(\w+)\.get\((\d+)\)\.ok_or\([^)]+\)\?, ([^)]+)\)',
        r'atomicAdd(&\1[\2], \3)',
        content
    )
    
    # Pattern 6: Simple assignments
    content = re.sub(
        r'= (\w+)\.get\((\d+)\)\.ok_or\([^)]+\)\?',
        r'= \1[\2]',
        content
    )
    
    # Pattern 7: Array assertions in tests
    content = re.sub(
        r'assert_eq!\(([^,]+)\.get\((\d+)\)\.ok_or\([^)]+\)\?, ([^)]+)\)',
        r'assert_eq!(\1[\2], \3)',
        content
    )
    
    # Pattern 8: String literals that shouldn't be in arrays
    content = re.sub(
        r'"(.+?)\.get\((\d+)\)\.ok_or\([^)]+\)\?"',
        r'"\1[\2]"',
        content
    )
    
    return content

def process_file(filepath):
    """Process a single file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Skip if no pattern found
        if '.get(' not in content or '.ok_or(' not in content:
            return False
        
        # Apply fixes
        content = fix_array_indexing(content)
        
        # Write back if changed
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    synthesis_dir = Path('/home/ubuntu/projects/stratoswarm/crates/synthesis/src')
    
    if not synthesis_dir.exists():
        print(f"Directory not found: {synthesis_dir}")
        sys.exit(1)
    
    files_processed = 0
    total_files = 0
    
    for filepath in synthesis_dir.rglob('*.rs'):
        total_files += 1
        if process_file(filepath):
            files_processed += 1
            print(f"Fixed: {filepath.relative_to(synthesis_dir)}")
    
    print(f"\nProcessed {files_processed} out of {total_files} files")

if __name__ == '__main__':
    main()