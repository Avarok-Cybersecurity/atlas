#!/bin/bash
# Script to count files recursively in /workspace/atlas/crates/

TARGET_DIR="/workspace/atlas/crates/"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist."
    exit 1
fi

# Count all files recursively
FILE_COUNT=$(find "$TARGET_DIR" -type f | wc -l)

echo "Total files in $TARGET_DIR: $FILE_COUNT"