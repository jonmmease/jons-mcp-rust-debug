#!/bin/bash
# Clean up temporary test and diagnostic files

echo "Cleaning up temporary files..."

# Remove test files
rm -f test_*.py
rm -f debug_*.py
rm -f diagnose_*.py
rm -f analyze_*.py
rm -f fix_*.py
rm -f verify_*.py
rm -f check_*.py
rm -f quick_*.py
rm -f add_*.py

# Keep important documentation
echo "Keeping documentation files:"
ls -la *.md

echo "Cleanup complete!"