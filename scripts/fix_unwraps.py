#!/usr/bin/env python3
"""
Automated script to help replace .unwrap() calls with proper error handling.
This script identifies unwrap() patterns and suggests replacements.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
from dataclasses import dataclass
from enum import Enum

class UnwrapContext(Enum):
    """Context where unwrap is used"""
    TEST = "test"           # In test code - might be acceptable
    MAIN = "main"           # In main function - needs proper handling
    FUNCTION = "function"   # In regular function - use ? operator
    CLOSURE = "closure"     # In closure - needs careful handling
    CONSTANT = "constant"   # In const/static context - might be acceptable

@dataclass
class UnwrapInstance:
    """Represents a single unwrap() occurrence"""
    file_path: str
    line_number: int
    line_content: str
    context: UnwrapContext
    suggested_fix: Optional[str]

class UnwrapFixer:
    """Analyzes and suggests fixes for unwrap() calls"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.unwrap_pattern = re.compile(r'\.unwrap\(\)')
        self.test_file_pattern = re.compile(r'(test|tests|benches?)/')
        self.test_attr_pattern = re.compile(r'#\[(cfg\()?test')
        
    def find_unwraps(self, file_path: Path) -> List[UnwrapInstance]:
        """Find all unwrap() calls in a file"""
        instances = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            in_test_context = self._is_test_file(file_path)
            
            for i, line in enumerate(lines, 1):
                if self.unwrap_pattern.search(line):
                    context = self._determine_context(lines, i-1, in_test_context)
                    suggested_fix = self._suggest_fix(line, context)
                    
                    instances.append(UnwrapInstance(
                        file_path=str(file_path),
                        line_number=i,
                        line_content=line.strip(),
                        context=context,
                        suggested_fix=suggested_fix
                    ))
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            
        return instances
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file"""
        return bool(self.test_file_pattern.search(str(file_path)))
    
    def _determine_context(self, lines: List[str], line_idx: int, in_test_file: bool) -> UnwrapContext:
        """Determine the context of the unwrap() call"""
        # Look backwards for context clues
        for i in range(max(0, line_idx - 20), line_idx):
            line = lines[i]
            
            if '#[test]' in line or '#[cfg(test)]' in line:
                return UnwrapContext.TEST
            if 'fn main(' in line:
                return UnwrapContext.MAIN
            if 'const ' in line or 'static ' in line:
                return UnwrapContext.CONSTANT
            if '|' in line and '}' in lines[line_idx]:  # Simple closure detection
                return UnwrapContext.CLOSURE
                
        if in_test_file:
            return UnwrapContext.TEST
            
        return UnwrapContext.FUNCTION
    
    def _suggest_fix(self, line: str, context: UnwrapContext) -> Optional[str]:
        """Suggest a fix for the unwrap() call"""
        if context == UnwrapContext.TEST:
            # In tests, unwrap might be acceptable but expect() is better
            return line.replace('.unwrap()', '.expect("test assertion failed")')
        
        elif context == UnwrapContext.MAIN:
            # In main, use expect with descriptive message
            return line.replace('.unwrap()', '.expect("Failed to initialize")')
        
        elif context == UnwrapContext.FUNCTION:
            # In functions, use ? operator
            return line.replace('.unwrap()', '?')
        
        elif context == UnwrapContext.CLOSURE:
            # In closures, might need different handling
            return line.replace('.unwrap()', '.expect("closure operation failed")')
        
        elif context == UnwrapContext.CONSTANT:
            # In const context, might need compile-time handling
            return None
        
        return None
    
    def analyze_crate(self, crate_path: Path) -> Tuple[List[UnwrapInstance], dict]:
        """Analyze all Rust files in a crate"""
        all_instances = []
        stats = {
            'total_files': 0,
            'files_with_unwraps': 0,
            'total_unwraps': 0,
            'by_context': {}
        }
        
        for rust_file in crate_path.rglob('*.rs'):
            stats['total_files'] += 1
            instances = self.find_unwraps(rust_file)
            
            if instances:
                stats['files_with_unwraps'] += 1
                stats['total_unwraps'] += len(instances)
                all_instances.extend(instances)
                
                for instance in instances:
                    context_name = instance.context.value
                    stats['by_context'][context_name] = stats['by_context'].get(context_name, 0) + 1
        
        return all_instances, stats
    
    def generate_fix_script(self, instances: List[UnwrapInstance], output_file: str):
        """Generate a shell script to apply fixes"""
        with open(output_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Automated unwrap() fix script\n")
            f.write("# Review each change before applying!\n\n")
            
            # Group by file
            by_file = {}
            for instance in instances:
                if instance.suggested_fix and instance.context != UnwrapContext.TEST:
                    if instance.file_path not in by_file:
                        by_file[instance.file_path] = []
                    by_file[instance.file_path].append(instance)
            
            for file_path, file_instances in by_file.items():
                f.write(f"\n# File: {file_path}\n")
                f.write(f"# {len(file_instances)} unwrap() calls to fix\n")
                
                # Sort by line number in reverse to avoid offset issues
                file_instances.sort(key=lambda x: x.line_number, reverse=True)
                
                for instance in file_instances:
                    f.write(f"\n# Line {instance.line_number}: {instance.context.value} context\n")
                    f.write(f"# Original: {instance.line_content}\n")
                    if instance.suggested_fix:
                        # Use sed for simple replacements
                        escaped_original = instance.line_content.replace('/', '\\/')
                        escaped_fix = instance.suggested_fix.replace('/', '\\/')
                        f.write(f"sed -i '{instance.line_number}s/{escaped_original}/{escaped_fix}/' {file_path}\n")
        
        os.chmod(output_file, 0o755)
        print(f"Fix script generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze and fix unwrap() calls')
    parser.add_argument('path', help='Path to crate or file to analyze')
    parser.add_argument('--fix', action='store_true', help='Generate fix script')
    parser.add_argument('--output', default='fix_unwraps.sh', help='Output script name')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    
    args = parser.parse_args()
    
    fixer = UnwrapFixer(os.getcwd())
    target_path = Path(args.path)
    
    if target_path.is_file():
        instances = fixer.find_unwraps(target_path)
        stats = {
            'total_files': 1,
            'files_with_unwraps': 1 if instances else 0,
            'total_unwraps': len(instances),
            'by_context': {}
        }
        for instance in instances:
            context_name = instance.context.value
            stats['by_context'][context_name] = stats['by_context'].get(context_name, 0) + 1
    else:
        instances, stats = fixer.analyze_crate(target_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Unwrap() Analysis Report for: {args.path}")
    print(f"{'='*60}")
    print(f"Total files analyzed: {stats['total_files']}")
    print(f"Files with unwrap(): {stats['files_with_unwraps']}")
    print(f"Total unwrap() calls: {stats['total_unwraps']}")
    
    if stats['by_context']:
        print(f"\nBy context:")
        for context, count in sorted(stats['by_context'].items()):
            print(f"  {context:12} : {count:5} occurrences")
    
    if not args.summary and instances:
        print(f"\n{'='*60}")
        print("Detailed findings:")
        print(f"{'='*60}")
        
        # Show first 10 non-test unwraps
        non_test = [i for i in instances if i.context != UnwrapContext.TEST][:10]
        for instance in non_test:
            print(f"\n{instance.file_path}:{instance.line_number}")
            print(f"  Context: {instance.context.value}")
            print(f"  Line: {instance.line_content}")
            if instance.suggested_fix:
                print(f"  Fix: {instance.suggested_fix}")
    
    if args.fix and instances:
        fixer.generate_fix_script(instances, args.output)
        print(f"\nTo apply fixes, review and run: ./{args.output}")
    
    # Exit with error code if unwraps found in non-test code
    non_test_count = sum(1 for i in instances if i.context != UnwrapContext.TEST)
    if non_test_count > 0:
        print(f"\n⚠️  Found {non_test_count} unwrap() calls in production code")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())