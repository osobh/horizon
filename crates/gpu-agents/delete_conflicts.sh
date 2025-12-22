#!/bin/bash
# Delete conflicting module files

cd /home/osobh/projects/exorust/crates/gpu-agents/src

# Try to delete the files
rm -f evolution.rs
rm -f knowledge.rs

# If that doesn't work, try moving them
mv evolution.rs evolution.rs.deleted 2>/dev/null || true
mv knowledge.rs knowledge.rs.deleted 2>/dev/null || true

# Create marker files to indicate deletion
echo "DELETED" > .evolution.rs.deleted
echo "DELETED" > .knowledge.rs.deleted

echo "Attempted to remove conflicting files"
ls -la | grep -E "(evolution|knowledge)\.rs"