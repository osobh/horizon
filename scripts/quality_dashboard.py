#!/usr/bin/env python3
"""
StratoSwarm Quality Metrics Dashboard

Provides real-time quality metrics for the StratoSwarm codebase following
the Rust Code Quality Standards.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

@dataclass
class QualityMetrics:
    """Quality metrics for the codebase"""
    total_lines: int = 0
    rust_files: int = 0
    test_files: int = 0
    unwrap_count: int = 0
    todo_count: int = 0
    unimplemented_count: int = 0
    unsafe_blocks: int = 0
    large_files: List[Tuple[str, int]] = None
    missing_docs: List[str] = None
    test_coverage: Optional[float] = None
    crate_count: int = 0
    
    def __post_init__(self):
        if self.large_files is None:
            self.large_files = []
        if self.missing_docs is None:
            self.missing_docs = []

class QualityDashboard:
    """Dashboard for monitoring code quality metrics"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.crates_dir = project_root / "crates"
        
    def collect_metrics(self) -> QualityMetrics:
        """Collect all quality metrics"""
        metrics = QualityMetrics()
        
        # Count crates
        if self.crates_dir.exists():
            metrics.crate_count = len(list(self.crates_dir.iterdir()))
        
        # Analyze all Rust files
        for rust_file in self.project_root.rglob("*.rs"):
            # Skip target and vendor directories
            if "target" in rust_file.parts or "vendor" in rust_file.parts:
                continue
                
            metrics.rust_files += 1
            
            # Check if it's a test file
            if rust_file.parent.name == "tests" or rust_file.name.endswith("_test.rs"):
                metrics.test_files += 1
            
            try:
                content = rust_file.read_text()
                lines = content.splitlines()
                metrics.total_lines += len(lines)
                
                # Count quality issues
                metrics.unwrap_count += content.count(".unwrap()")
                metrics.todo_count += content.count("todo!(")
                metrics.unimplemented_count += content.count("unimplemented!(")
                metrics.unsafe_blocks += content.count("unsafe {")
                
                # Check for large files
                if len(lines) > 850:
                    relative_path = rust_file.relative_to(self.project_root)
                    metrics.large_files.append((str(relative_path), len(lines)))
                
                # Check for missing documentation
                if rust_file.name == "lib.rs" or rust_file.name == "main.rs":
                    if not content.startswith("//!"):
                        relative_path = rust_file.relative_to(self.project_root)
                        metrics.missing_docs.append(str(relative_path))
                        
            except Exception as e:
                print(f"Error analyzing {rust_file}: {e}")
        
        # Sort large files by line count
        metrics.large_files.sort(key=lambda x: x[1], reverse=True)
        
        return metrics
    
    def check_git_status(self) -> Dict[str, int]:
        """Check git status for uncommitted changes"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                modified = sum(1 for line in lines if line.startswith(' M'))
                untracked = sum(1 for line in lines if line.startswith('??'))
                staged = sum(1 for line in lines if line[0] in 'AM')
                
                return {
                    'modified': modified,
                    'untracked': untracked,
                    'staged': staged,
                    'total': len(lines)
                }
        except Exception:
            return {'modified': 0, 'untracked': 0, 'staged': 0, 'total': 0}
    
    def format_number(self, num: int, threshold: int = 0, reverse: bool = False) -> str:
        """Format number with color based on threshold"""
        if threshold == 0:
            return str(num)
        
        if reverse:
            # Lower is better (e.g., unwrap count)
            if num == 0:
                return f"{Colors.GREEN}{num}{Colors.RESET}"
            elif num <= threshold:
                return f"{Colors.YELLOW}{num}{Colors.RESET}"
            else:
                return f"{Colors.RED}{num}{Colors.RESET}"
        else:
            # Higher is better (e.g., test coverage)
            if num >= threshold:
                return f"{Colors.GREEN}{num}{Colors.RESET}"
            else:
                return f"{Colors.RED}{num}{Colors.RESET}"
    
    def print_dashboard(self, metrics: QualityMetrics):
        """Print the quality dashboard"""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}   StratoSwarm Quality Metrics Dashboard{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
        
        # Project Overview
        print(f"{Colors.BOLD}📊 Project Overview{Colors.RESET}")
        print(f"   Total Crates:     {metrics.crate_count}")
        print(f"   Rust Files:       {metrics.rust_files}")
        print(f"   Test Files:       {metrics.test_files}")
        print(f"   Total Lines:      {metrics.total_lines:,}")
        print()
        
        # Quality Metrics
        print(f"{Colors.BOLD}✨ Quality Metrics{Colors.RESET}")
        
        unwrap_status = "✅" if metrics.unwrap_count == 0 else "⚠️" if metrics.unwrap_count < 100 else "❌"
        print(f"   {unwrap_status} unwrap() calls:     {self.format_number(metrics.unwrap_count, 100, reverse=True)}")
        
        todo_status = "✅" if metrics.todo_count == 0 else "❌"
        print(f"   {todo_status} todo!() macros:     {self.format_number(metrics.todo_count, 0, reverse=True)}")
        
        unimpl_status = "✅" if metrics.unimplemented_count == 0 else "❌"
        print(f"   {unimpl_status} unimplemented!():   {self.format_number(metrics.unimplemented_count, 0, reverse=True)}")
        
        unsafe_status = "✅" if metrics.unsafe_blocks == 0 else "⚠️"
        print(f"   {unsafe_status} unsafe blocks:      {self.format_number(metrics.unsafe_blocks, 10, reverse=True)}")
        print()
        
        # File Size Analysis
        print(f"{Colors.BOLD}📏 File Size Analysis{Colors.RESET}")
        large_file_status = "✅" if len(metrics.large_files) == 0 else "⚠️"
        print(f"   {large_file_status} Files > 850 lines:  {len(metrics.large_files)}")
        
        if metrics.large_files:
            print(f"\n   {Colors.YELLOW}Large files requiring split:{Colors.RESET}")
            for file_path, line_count in metrics.large_files[:5]:
                print(f"     • {file_path} ({line_count} lines)")
            if len(metrics.large_files) > 5:
                print(f"     ... and {len(metrics.large_files) - 5} more")
        print()
        
        # Documentation Status
        print(f"{Colors.BOLD}📚 Documentation Status{Colors.RESET}")
        doc_status = "✅" if len(metrics.missing_docs) == 0 else "⚠️"
        print(f"   {doc_status} Files missing docs: {len(metrics.missing_docs)}")
        
        if metrics.missing_docs:
            print(f"\n   {Colors.YELLOW}Files needing documentation:{Colors.RESET}")
            for file_path in metrics.missing_docs[:5]:
                print(f"     • {file_path}")
            if len(metrics.missing_docs) > 5:
                print(f"     ... and {len(metrics.missing_docs) - 5} more")
        print()
        
        # Git Status
        git_status = self.check_git_status()
        if git_status['total'] > 0:
            print(f"{Colors.BOLD}🔄 Git Status{Colors.RESET}")
            print(f"   Modified files:    {git_status['modified']}")
            print(f"   Untracked files:   {git_status['untracked']}")
            print(f"   Staged files:      {git_status['staged']}")
            print()
        
        # Quality Score
        quality_score = self.calculate_quality_score(metrics)
        score_color = Colors.GREEN if quality_score >= 80 else Colors.YELLOW if quality_score >= 60 else Colors.RED
        
        print(f"{Colors.BOLD}🎯 Overall Quality Score{Colors.RESET}")
        print(f"   {score_color}{Colors.BOLD}{quality_score:.1f}/100{Colors.RESET}")
        
        # Progress bars
        self.print_progress_bar("Quality", quality_score, 100)
        
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}\n")
        
        # Recommendations
        if quality_score < 100:
            self.print_recommendations(metrics)
    
    def calculate_quality_score(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score (0-100)"""
        score = 100.0
        
        # Deduct points for quality issues
        if metrics.unwrap_count > 0:
            score -= min(30, metrics.unwrap_count / 100)  # Max 30 point deduction
        
        if metrics.todo_count > 0:
            score -= min(10, metrics.todo_count * 2)  # Max 10 point deduction
        
        if metrics.unimplemented_count > 0:
            score -= min(10, metrics.unimplemented_count * 2)  # Max 10 point deduction
        
        if metrics.unsafe_blocks > 0:
            score -= min(5, metrics.unsafe_blocks / 2)  # Max 5 point deduction
        
        if len(metrics.large_files) > 0:
            score -= min(15, len(metrics.large_files) * 1.5)  # Max 15 point deduction
        
        if len(metrics.missing_docs) > 0:
            score -= min(10, len(metrics.missing_docs))  # Max 10 point deduction
        
        # Bonus for test files
        if metrics.rust_files > 0:
            test_ratio = metrics.test_files / metrics.rust_files
            if test_ratio > 0.3:
                score += min(10, test_ratio * 20)  # Max 10 point bonus
        
        return max(0, min(100, score))
    
    def print_progress_bar(self, label: str, value: float, max_value: float, width: int = 40):
        """Print a progress bar"""
        percentage = value / max_value
        filled = int(width * percentage)
        bar = '█' * filled + '░' * (width - filled)
        
        color = Colors.GREEN if percentage >= 0.8 else Colors.YELLOW if percentage >= 0.6 else Colors.RED
        print(f"   {label}: {color}{bar}{Colors.RESET} {value:.1f}%")
    
    def print_recommendations(self, metrics: QualityMetrics):
        """Print improvement recommendations"""
        print(f"{Colors.BOLD}💡 Recommendations for Improvement{Colors.RESET}\n")
        
        if metrics.unwrap_count > 0:
            print(f"   {Colors.YELLOW}•{Colors.RESET} Remove {metrics.unwrap_count} unwrap() calls")
            print(f"     Run: python scripts/fix_unwraps.py --auto-fix\n")
        
        if metrics.todo_count > 0:
            print(f"   {Colors.YELLOW}•{Colors.RESET} Implement {metrics.todo_count} todo!() macros\n")
        
        if len(metrics.large_files) > 0:
            print(f"   {Colors.YELLOW}•{Colors.RESET} Split {len(metrics.large_files)} large files (>850 lines)")
            print(f"     Largest: {metrics.large_files[0][0]} ({metrics.large_files[0][1]} lines)\n")
        
        if len(metrics.missing_docs) > 0:
            print(f"   {Colors.YELLOW}•{Colors.RESET} Add documentation to {len(metrics.missing_docs)} files\n")
        
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")

def main():
    """Main entry point"""
    # Find project root
    current_dir = Path.cwd()
    project_root = current_dir
    
    # Search for project root (containing Cargo.toml)
    while project_root.parent != project_root:
        if (project_root / "Cargo.toml").exists():
            break
        project_root = project_root.parent
    
    if not (project_root / "Cargo.toml").exists():
        print(f"{Colors.RED}Error: Could not find project root (no Cargo.toml found){Colors.RESET}")
        sys.exit(1)
    
    dashboard = QualityDashboard(project_root)
    
    print(f"{Colors.BLUE}Analyzing StratoSwarm codebase...{Colors.RESET}")
    metrics = dashboard.collect_metrics()
    
    dashboard.print_dashboard(metrics)
    
    # Exit with non-zero if quality score is below threshold
    quality_score = dashboard.calculate_quality_score(metrics)
    if quality_score < 60:
        sys.exit(1)

if __name__ == "__main__":
    main()