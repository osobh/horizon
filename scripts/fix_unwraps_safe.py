#!/usr/bin/env python3
"""
Safe unwrap() fixer that makes targeted replacements
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

def fix_unwrap_in_file(file_path: Path, dry_run: bool = True) -> List[Tuple[int, str, str]]:
    """Fix unwrap() calls in a single file"""
    
    if not file_path.exists():
        return []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    changes = []
    modified_lines = []
    
    for i, line in enumerate(lines):
        original_line = line
        line_num = i + 1
        
        # Skip test files and test functions
        if '#[test]' in ''.join(lines[max(0, i-5):i]) or \
           '#[tokio::test]' in ''.join(lines[max(0, i-5):i]) or \
           'mod tests' in ''.join(lines[max(0, i-2):i]):
            modified_lines.append(original_line)
            continue
        
        # Pattern 1: .unwrap() at end of line -> ?
        if '.unwrap()' in line and not line.strip().startswith('//'):
            # Check if we're in a Result-returning context
            if any(kw in ''.join(lines[max(0, i-10):i]) for kw in ['-> Result', '-> anyhow::Result', 'fn ', 'async fn']):
                new_line = line.replace('.unwrap()', '?')
                if new_line != line:
                    changes.append((line_num, line.strip(), new_line.strip()))
                    line = new_line
        
        # Pattern 2: .unwrap(); -> ?;
        if '.unwrap();' in line and not line.strip().startswith('//'):
            if any(kw in ''.join(lines[max(0, i-10):i]) for kw in ['-> Result', '-> anyhow::Result']):
                new_line = line.replace('.unwrap();', '?;')
                if new_line != line:
                    changes.append((line_num, line.strip(), new_line.strip()))
                    line = new_line
        
        # Pattern 3: unwrap_or_else(|| panic!(...)) -> expect(...)
        panic_pattern = r'\.unwrap_or_else\(\|\|\s*panic!\((.*?)\)\)'
        if re.search(panic_pattern, line):
            new_line = re.sub(panic_pattern, r'.expect(\1)', line)
            if new_line != line:
                changes.append((line_num, line.strip(), new_line.strip()))
                line = new_line
        
        modified_lines.append(line)
    
    if not dry_run and changes:
        with open(file_path, 'w') as f:
            f.writelines(modified_lines)
    
    return changes

def process_crate(crate_path: Path, dry_run: bool = True) -> None:
    """Process all Rust files in a crate"""
    
    total_changes = 0
    files_modified = 0
    
    for rust_file in crate_path.rglob("*.rs"):
        # Skip target directory
        if "target" in rust_file.parts:
            continue
        
        changes = fix_unwrap_in_file(rust_file, dry_run)
        
        if changes:
            files_modified += 1
            total_changes += len(changes)
            
            print(f"\n{rust_file.relative_to(crate_path)}:")
            for line_num, old, new in changes[:5]:  # Show first 5 changes
                print(f"  Line {line_num}:")
                print(f"    - {old}")
                print(f"    + {new}")
            
            if len(changes) > 5:
                print(f"  ... and {len(changes) - 5} more changes")
    
    print(f"\n{'='*60}")
    print(f"Summary for {crate_path.name}:")
    print(f"  Files to modify: {files_modified}")
    print(f"  Total unwrap() calls to fix: {total_changes}")
    
    if dry_run:
        print(f"\nThis was a dry run. To apply changes, use --apply")
    else:
        print(f"\n✅ Changes applied successfully!")

def main():
    parser = argparse.ArgumentParser(description='Safe unwrap() fixer')
    parser.add_argument('path', help='Path to crate or file')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default is dry run)')
    
    args = parser.parse_args()
    path = Path(args.path)
    
    if path.is_file():
        changes = fix_unwrap_in_file(path, dry_run=not args.apply)
        if changes:
            print(f"Found {len(changes)} unwrap() calls to fix")
            for line_num, old, new in changes:
                print(f"  Line {line_num}:")
                print(f"    - {old}")
                print(f"    + {new}")
        else:
            print("No unwrap() calls to fix")
    elif path.is_dir():
        process_crate(path, dry_run=not args.apply)
    else:
        print(f"Error: {path} not found")
        return 1

if __name__ == "__main__":
    main()