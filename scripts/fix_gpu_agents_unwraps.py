#!/usr/bin/env python3
"""
Fix unwrap() calls in gpu-agents crate
"""

import os
import re
import sys
from pathlib import Path

def fix_unwrap_patterns(content):
    """Fix common unwrap patterns in gpu-agents code"""
    
    # Pattern 1: CudaDevice::new(0).unwrap() -> CudaDevice::new(0)?
    content = re.sub(
        r'CudaDevice::new\((\d+)\)\.unwrap\(\)',
        r'CudaDevice::new(\1)?',
        content
    )
    
    # Pattern 2: Arc::new(CudaDevice::new(0).unwrap()) -> Arc::new(CudaDevice::new(0)?)
    content = re.sub(
        r'Arc::new\(CudaDevice::new\((\d+)\)\.unwrap\(\)\)',
        r'Arc::new(CudaDevice::new(\1)?)',
        content
    )
    
    # Pattern 3: .await.unwrap() -> .await?
    content = re.sub(
        r'\.await\.unwrap\(\)',
        r'.await?',
        content
    )
    
    # Pattern 4: something.unwrap() in test functions -> keep as is
    # Don't modify unwrap in test functions
    
    # Pattern 5: let x = y.unwrap(); -> let x = y?;
    content = re.sub(
        r'let (\w+) = (.+?)\.unwrap\(\);',
        r'let \1 = \2?;',
        content
    )
    
    # Pattern 6: foo().unwrap() -> foo()?
    content = re.sub(
        r'(\w+\([^)]*\))\.unwrap\(\)',
        r'\1?',
        content
    )
    
    return content

def should_process_file(filepath):
    """Check if file should be processed"""
    # Skip test files
    if 'test' in filepath.name.lower() or filepath.suffix != '.rs':
        return False
    # Skip bin files (they often have different error handling)
    if '/bin/' in str(filepath):
        return False
    return True

def fix_function_signatures(content):
    """Fix function signatures to return Result where needed"""
    
    # Find functions that use ? operator but don't return Result
    lines = content.split('\n')
    in_function = False
    function_start = -1
    modified_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for function definition
        if re.match(r'^\s*(pub\s+)?(async\s+)?fn\s+\w+', line):
            # Check if it already returns Result
            if 'Result<' not in line and '->' not in line:
                # Look ahead to see if function body uses ?
                j = i + 1
                uses_question_mark = False
                brace_count = 0
                while j < len(lines) and j < i + 50:  # Check next 50 lines
                    if '{' in lines[j]:
                        brace_count += lines[j].count('{')
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                    if '?' in lines[j] and not '//' in lines[j]:
                        uses_question_mark = True
                    if brace_count == 0 and j > i:
                        break
                    j += 1
                
                if uses_question_mark:
                    # Add Result return type
                    if '(' in line and ')' in line:
                        before_paren = line.split(')')[0] + ')'
                        after_paren = ')'.join(line.split(')')[1:])
                        if '{' in after_paren:
                            line = before_paren + ' -> Result<(), Box<dyn std::error::Error>> ' + after_paren
                        else:
                            line = before_paren + ' -> Result<(), Box<dyn std::error::Error>>'
        
        modified_lines.append(line)
        i += 1
    
    return '\n'.join(modified_lines)

def process_file(filepath):
    """Process a single file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Skip if no unwrap() calls
        if '.unwrap()' not in content:
            return False
        
        # Apply fixes
        content = fix_unwrap_patterns(content)
        
        # Fix function signatures if needed
        if '?' in content:
            content = fix_function_signatures(content)
        
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
    gpu_agents_dir = Path('/home/ubuntu/projects/stratoswarm/crates/gpu-agents/src')
    
    if not gpu_agents_dir.exists():
        print(f"Directory not found: {gpu_agents_dir}")
        sys.exit(1)
    
    files_processed = 0
    total_files = 0
    
    for filepath in gpu_agents_dir.rglob('*.rs'):
        if should_process_file(filepath):
            total_files += 1
            if process_file(filepath):
                files_processed += 1
                print(f"Fixed: {filepath.relative_to(gpu_agents_dir)}")
    
    print(f"\nProcessed {files_processed} out of {total_files} files")
    
    # Count remaining unwraps
    remaining = 0
    for filepath in gpu_agents_dir.rglob('*.rs'):
        if should_process_file(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
                remaining += content.count('.unwrap()')
    
    print(f"Remaining unwrap() calls in non-test files: {remaining}")

if __name__ == '__main__':
    main()