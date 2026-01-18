#!/bin/bash

################################################################################
# Generalized ROOT File Merger Script
# 
# Purpose: Merge ROOT files from subdirectories using hadd
# Usage: ./merging_rootfiles_locally.sh [SOURCE_DIR] [FILE_PATTERN] [OUTPUT_DIR] [LOG_FILE]
#
# Examples:
#   ./merging_rootfiles_locally.sh 1275 "GCo_320[0134].root" . log1.log
#   ./merging_rootfiles_locally.sh 2412 "GCo_*.root" merged_output merge.log
#
# Variables can be set inside the script or passed as arguments
################################################################################

# ============================================================================
# CONFIGURABLE VARIABLES - Edit these to change default behavior
# You can also override by passing command-line arguments
# ============================================================================

# Option 1: Set variables directly in the script (uncomment and modify)
SOURCE_DIR="2414"
FILE_PATTERN="GCo_340[0134].root"
OUTPUT_DIR="./2414"
LOG_FILE="$OUTPUT_DIR/log_merge.log"

# Option 2: Use command-line arguments (default)
#SOURCE_DIR="${1:-.}"
#FILE_PATTERN="${2:-GCo_*.root}"
#OUTPUT_DIR="${3:-.}"
#LOG_FILE="${4:-merge.log}"

# ============================================================================

# Validation
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "ERROR: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: Output directory '$OUTPUT_DIR' does not exist"
    exit 1
fi

# Initialize log file
echo "========================================" >> "$LOG_FILE"
echo "Merge Job Started: $(date)" >> "$LOG_FILE"
echo "Source Directory: $SOURCE_DIR" >> "$LOG_FILE"
echo "File Pattern: $FILE_PATTERN" >> "$LOG_FILE"
echo "Output Directory: $OUTPUT_DIR" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

echo "Configuration:"
echo "  Source Directory: $SOURCE_DIR"
echo "  File Pattern: $FILE_PATTERN"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Log File: $LOG_FILE"
echo ""

# Find all unique file names matching pattern in subdirectories (recursive, unlimited depth)
UNIQUE_FILES=$(find "$SOURCE_DIR" -name "$FILE_PATTERN" | xargs -n1 basename | sort -u)

if [[ -z "$UNIQUE_FILES" ]]; then
    echo "WARNING: No files matching pattern '$FILE_PATTERN' found in '$SOURCE_DIR'" | tee -a "$LOG_FILE"
    exit 1
fi

# Merge each unique file
FILE_COUNT=0
for FILENAME in $UNIQUE_FILES; do
    OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
    SOURCE_FILES=$(find "$SOURCE_DIR" -name "$FILENAME")
    
    if [[ -z "$SOURCE_FILES" ]]; then
        echo "WARNING: No source files found for $FILENAME" | tee -a "$LOG_FILE"
        continue
    fi
    
    echo "Processing: $FILENAME" | tee -a "$LOG_FILE"
    echo "  Source files: $(echo "$SOURCE_FILES" | wc -l)" | tee -a "$LOG_FILE"
    echo "  Output: $OUTPUT_FILE" | tee -a "$LOG_FILE"
    
    # Run hadd with output redirected to log
    hadd "$OUTPUT_FILE" $SOURCE_FILES &>> "$LOG_FILE"
    
    if [[ $? -eq 0 ]]; then
        echo "  ✓ SUCCESS" | tee -a "$LOG_FILE"
        ((FILE_COUNT++))
    else
        echo "  ✗ FAILED" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "Merge Job Completed: $(date)" >> "$LOG_FILE"
echo "Files Merged: $FILE_COUNT" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

echo "Merge complete. Results: $FILE_COUNT files merged. See $LOG_FILE for details."
