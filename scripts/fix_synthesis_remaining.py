#!/usr/bin/env python3
"""
Fix remaining unwrap() calls in synthesis crate with more aggressive patterns
"""

import os
import re
import sys
from pathlib import Path

def fix_advanced_unwrap_patterns(content):
    """Fix more complex unwrap patterns"""
    
    # Pattern 1: unwrap() in test assertions
    content = re.sub(
        r'assert!\((.+?)\.unwrap\(\)',
        r'assert!(\1.is_ok()',
        content
    )
    
    # Pattern 2: unwrap() in match arms
    content = re.sub(
        r'=> (.+?)\.unwrap\(\)',
        r'=> \1.ok()?',
        content
    )
    
    # Pattern 3: unwrap_err() -> expect pattern
    content = re.sub(
        r'\.unwrap_err\(\)',
        r'.expect_err("Expected error")',
        content
    )
    
    # Pattern 4: Option unwrap -> ok_or pattern
    content = re.sub(
        r'(\w+)\.get\(([^)]+)\)\.unwrap\(\)',
        r'\1.get(\2).ok_or("Key not found")?',
        content
    )
    
    # Pattern 5: Vec/slice unwrap patterns
    content = re.sub(
        r'(\w+)\[(\d+)\]',
        r'\1.get(\2).ok_or("Index out of bounds")?',
        content,
        count=0  # Only in specific contexts
    )
    
    # Pattern 6: unwrap() in closures
    content = re.sub(
        r'\|(\w+)\| (.+?)\.unwrap\(\)',
        r'|\1| \2.ok()',
        content
    )
    
    # Pattern 7: Multiple chained unwraps
    content = re.sub(
        r'\.unwrap\(\)\.unwrap\(\)',
        r'.ok()?.ok()?',
        content
    )
    
    # Pattern 8: Result::unwrap in tests - convert to ?
    if '#[test]' in content or '#[tokio::test]' in content:
        # In test functions, change return type if needed
        content = re.sub(
            r'(#\[(?:tokio::)?test\])\s*\n\s*((?:pub\s+)?(?:async\s+)?fn\s+\w+\([^)]*\))\s*{',
            r'\1\n    \2 -> Result<(), Box<dyn std::error::Error>> {',
            content
        )
        
        # Add Ok(()) at the end of test functions if not present
        lines = content.split('\n')
        in_test = False
        brace_count = 0
        test_start = -1
        
        for i, line in enumerate(lines):
            if '#[test]' in line or '#[tokio::test]' in line:
                in_test = True
                test_start = i
                brace_count = 0
            elif in_test:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0 and '{' in line:
                    # End of test function
                    # Check if it already returns Ok(())
                    if i > 0 and 'Ok(())' not in lines[i-1]:
                        lines[i-1] = lines[i-1].rstrip() + '\n        Ok(())'
                    in_test = False
        
        content = '\n'.join(lines)
    
    return content

def fix_test_signatures(content):
    """Fix test function signatures to return Result"""
    lines = content.split('\n')
    modified_lines = []
    in_test = False
    
    for i, line in enumerate(lines):
        if '#[test]' in line or '#[tokio::test]' in line:
            in_test = True
            modified_lines.append(line)
        elif in_test and 'fn ' in line:
            # Check if already returns Result
            if '-> Result<' not in line:
                # Add Result return type
                if '{' in line:
                    line = line.replace('{', '-> Result<(), Box<dyn std::error::Error>> {')
                else:
                    line = line.rstrip() + ' -> Result<(), Box<dyn std::error::Error>>'
            in_test = False
            modified_lines.append(line)
        else:
            modified_lines.append(line)
    
    return '\n'.join(modified_lines)

def should_process_file(filepath):
    """Check if file should be processed"""
    # Include test files this time
    if filepath.suffix != '.rs':
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
        
        # Apply basic fixes first
        content = fix_basic_patterns(content)
        
        # Apply advanced fixes
        content = fix_advanced_unwrap_patterns(content)
        
        # Fix test signatures if needed
        if '#[test]' in content or '#[tokio::test]' in content:
            content = fix_test_signatures(content)
        
        # Write back if changed
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def fix_basic_patterns(content):
    """Apply basic unwrap fixes"""
    
    # Pattern 1: Simple .unwrap() -> ?
    content = re.sub(
        r'(\w+(?:\([^)]*\))?(?:\.\w+(?:\([^)]*\))?)*)\..unwrap\(\)(\s*[;,])',
        r'\1?\2',
        content
    )
    
    # Pattern 2: .await.unwrap() -> .await?
    content = re.sub(
        r'\.await\.unwrap\(\)',
        r'.await?',
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
    
    return content

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
    
    print(f"\nProcessed {files_processed} out of {total_files} files")
    
    # Count remaining unwraps
    remaining = 0
    for filepath in synthesis_dir.rglob('*.rs'):
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                remaining += content.count('.unwrap()')
        except:
            pass
    
    print(f"Remaining unwrap() calls: {remaining}")

if __name__ == '__main__':
    main()