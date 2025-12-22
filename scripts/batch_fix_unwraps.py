#!/usr/bin/env python3
"""
Batch fix unwrap() calls across multiple crates
"""

import os
import re
import sys
from pathlib import Path

def fix_unwrap_patterns(content):
    """Fix common unwrap patterns"""
    
    # Pattern 1: Simple .unwrap() -> ?
    content = re.sub(
        r'(\w+(?:\([^)]*\))?(?:\.\w+(?:\([^)]*\))?)*)\.unwrap\(\)(\s*[;,])',
        r'\1?\2',
        content
    )
    
    # Pattern 2: .await.unwrap() -> .await?
    content = re.sub(
        r'\.await\.unwrap\(\)',
        r'.await?',
        content
    )
    
    # Pattern 3: Arc::new(something.unwrap()) -> Arc::new(something?)
    content = re.sub(
        r'Arc::new\(([^)]+)\.unwrap\(\)\)',
        r'Arc::new(\1?)',
        content
    )
    
    # Pattern 4: .lock().unwrap() -> .lock().map_err(|e| format!("Lock error: {}", e))?
    content = re.sub(
        r'\.lock\(\)\.unwrap\(\)',
        r'.lock().map_err(|e| format!("Lock error: {}", e))?',
        content
    )
    
    # Pattern 5: .parse().unwrap() -> .parse()?
    content = re.sub(
        r'\.parse\(\)\.unwrap\(\)',
        r'.parse()?',
        content
    )
    
    return content

def should_process_file(filepath):
    """Check if file should be processed"""
    # Skip test files
    if 'test' in filepath.name.lower():
        return False
    if filepath.suffix != '.rs':
        return False
    if '/tests/' in str(filepath):
        return False
    if '/bin/' in str(filepath):
        return False
    return True

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
        
        # Write back if changed
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def process_crate(crate_name, crate_path):
    """Process a single crate"""
    print(f"\n=== Processing {crate_name} ===")
    
    if not crate_path.exists():
        print(f"  Path not found: {crate_path}")
        return 0, 0
    
    files_processed = 0
    total_files = 0
    
    for filepath in crate_path.rglob('*.rs'):
        if should_process_file(filepath):
            total_files += 1
            if process_file(filepath):
                files_processed += 1
                print(f"  Fixed: {filepath.relative_to(crate_path)}")
    
    # Count remaining unwraps
    remaining = 0
    for filepath in crate_path.rglob('*.rs'):
        if should_process_file(filepath):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    remaining += content.count('.unwrap()')
            except:
                pass
    
    print(f"  Processed {files_processed}/{total_files} files, {remaining} unwrap() remaining")
    return files_processed, remaining

def main():
    base_path = Path('/home/ubuntu/projects/stratoswarm/crates')
    
    # High-priority crates to fix
    crates_to_fix = [
        'governance',
        'fault-tolerance', 
        'storage',
        'compliance',
        'time-travel-debugger',
        'zero-trust',
        'evolution-engines'
    ]
    
    total_fixed = 0
    total_remaining = 0
    
    for crate_name in crates_to_fix:
        crate_path = base_path / crate_name / 'src'
        fixed, remaining = process_crate(crate_name, crate_path)
        total_fixed += fixed
        total_remaining += remaining
    
    print(f"\n=== SUMMARY ===")
    print(f"Total files fixed: {total_fixed}")
    print(f"Total unwrap() remaining: {total_remaining}")

if __name__ == '__main__':
    main()