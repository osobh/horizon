#!/usr/bin/env python3
"""
Fix unwrap() calls in synthesis crate (excluding test files)
"""

import os
import re
import sys
from pathlib import Path

def fix_unwrap_patterns(content):
    """Fix common unwrap patterns in synthesis code"""
    
    # Pattern 1: .unwrap() at end of line -> ?
    content = re.sub(
        r'(\w+(?:\([^)]*\))?(?:\.\w+(?:\([^)]*\))?)*)\.unwrap\(\)(\s*[;,])',
        r'\1?\2',
        content
    )
    
    # Pattern 2: let x = y.unwrap(); -> let x = y?;
    content = re.sub(
        r'let (\w+) = (.+?)\.unwrap\(\);',
        r'let \1 = \2?;',
        content
    )
    
    # Pattern 3: .parse().unwrap() -> .parse()?
    content = re.sub(
        r'\.parse\(\)\.unwrap\(\)',
        r'.parse()?',
        content
    )
    
    # Pattern 4: .lock().unwrap() -> .lock().map_err(...)?
    content = re.sub(
        r'\.lock\(\)\.unwrap\(\)',
        r'.lock().map_err(|e| format!("Lock error: {}", e))?',
        content
    )
    
    # Pattern 5: .join().unwrap() -> .join().map_err(...)?
    content = re.sub(
        r'\.join\(\)\.unwrap\(\)',
        r'.join().map_err(|_| "Thread join error")?',
        content
    )
    
    # Pattern 6: HashMap get unwrap -> get with error handling
    content = re.sub(
        r'(\w+)\.get\(([^)]+)\)\.unwrap\(\)',
        r'\1.get(\2).ok_or("Key not found")?',
        content
    )
    
    return content

def fix_function_signatures(content):
    """Fix function signatures to return Result where needed"""
    
    lines = content.split('\n')
    modified_lines = []
    
    for i, line in enumerate(lines):
        # Skip if already returns Result
        if 'Result<' in line:
            modified_lines.append(line)
            continue
            
        # Check for function definition that might need Result
        if re.match(r'^\s*(pub\s+)?(async\s+)?fn\s+\w+', line):
            # Look ahead to see if function uses ?
            uses_question = False
            for j in range(i+1, min(i+30, len(lines))):
                if '?' in lines[j] and '//' not in lines[j]:
                    uses_question = True
                    break
            
            if uses_question and '->' not in line and 'Result<' not in line:
                # Add Result return type
                if '{' in line:
                    line = line.replace('{', '-> Result<(), Box<dyn std::error::Error>> {')
                else:
                    line = line.rstrip() + ' -> Result<(), Box<dyn std::error::Error>>'
        
        modified_lines.append(line)
    
    return '\n'.join(modified_lines)

def should_process_file(filepath):
    """Check if file should be processed"""
    # Skip test files and generated files
    if 'test' in filepath.name.lower():
        return False
    if filepath.suffix != '.rs':
        return False
    if '/tests/' in str(filepath):
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
        
        # Fix function signatures if needed
        if '?' in content:
            content = fix_function_signatures(content)
        
        # Add Result import if needed and not present
        if '?' in content and 'use std::error::Error' not in content:
            # Add import at the beginning after other use statements
            lines = content.split('\n')
            import_added = False
            for i, line in enumerate(lines):
                if line.startswith('use ') and not import_added:
                    # Find the last use statement
                    j = i
                    while j < len(lines) and (lines[j].startswith('use ') or lines[j].strip() == ''):
                        j += 1
                    lines.insert(j, 'use std::error::Error;')
                    import_added = True
                    break
            content = '\n'.join(lines)
        
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
        if should_process_file(filepath):
            total_files += 1
            if process_file(filepath):
                files_processed += 1
                print(f"Fixed: {filepath.relative_to(synthesis_dir)}")
    
    print(f"\nProcessed {files_processed} out of {total_files} non-test files")
    
    # Count remaining unwraps
    remaining = 0
    for filepath in synthesis_dir.rglob('*.rs'):
        if should_process_file(filepath):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    remaining += content.count('.unwrap()')
            except:
                pass
    
    print(f"Remaining unwrap() calls in non-test files: {remaining}")

if __name__ == '__main__':
    main()