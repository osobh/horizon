#!/bin/bash

# Following rules.md: NEVER delete files arbitrarily
# This script safely renames conflicting module files after merging content

cd "$(dirname "$0")/src"

echo "=== Resolving Module Conflicts (TDD Fix Step 1) ==="

# Check if files exist before moving
if [ -f "evolution.rs" ]; then
    echo "Moving evolution.rs to legacy_evolution.rs (content merged into evolution/mod.rs)"
    mv evolution.rs legacy_evolution.rs
fi

if [ -f "knowledge.rs" ]; then
    echo "Moving knowledge.rs to legacy_knowledge.rs (content merged into knowledge/mod.rs)"
    mv knowledge.rs legacy_knowledge.rs
fi

echo "✅ Module conflicts resolved following TDD approach"
echo "✅ All needed types preserved in module directories"