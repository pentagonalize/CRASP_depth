#!/bin/bash

# Define the test ranges (modify this to change ranges for all processing)
# Format: "start1,end1;start2,end2;start3,end3;..."
TEST_RANGES="201,250;251,300;301,350;351,400"

# Define the output files
OUTPUT_CSV="combined_results.csv"
OUTPUT_LATEX="latex_tables_combined.tex"

# Function to show usage
show_usage() {
    echo "Usage: $0 [output_csv] [-r test_ranges] [-l output_latex] [--no-latex]"
    echo ""
    echo "Arguments:"
    echo "  output_csv     Output CSV file (default: combined_results.csv)"
    echo "  -r ranges      Test ranges in format 'start1,end1;start2,end2;...'"
    echo "  -l latex_file  Output LaTeX file (default: latex_tables_combined.tex)"
    echo "  --no-latex     Skip LaTeX table generation"
    echo ""
}

# Parse command line arguments
GENERATE_LATEX=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--ranges)
            TEST_RANGES="$2"
            shift 2
            ;;
        -l|--latex)
            OUTPUT_LATEX="$2"
            shift 2
            ;;
        --no-latex)
            GENERATE_LATEX=false
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            # Assume it's the output CSV file if not already set
            if [ "$OUTPUT_CSV" = "combined_results.csv" ]; then
                OUTPUT_CSV="$1"
            else
                echo "Error: Multiple non-option arguments provided"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

echo "Using test ranges: $TEST_RANGES"
echo "Output CSV file: $OUTPUT_CSV"
if [ "$GENERATE_LATEX" = true ]; then
    echo "Output LaTeX file: $OUTPUT_LATEX"
else
    echo "LaTeX generation: DISABLED"
fi

# Remove the output file if it exists (to start fresh)
rm -f "$OUTPUT_CSV"

# Flag to track if this is the first file (for header creation)
FIRST_FILE=true

# Count total files to process
TOTAL_FILES=$(find . -type f -name "training_log.txt" | wc -l)
CURRENT_FILE=0

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "No training_log.txt files found in current directory or subdirectories."
    exit 1
fi

echo "Found $TOTAL_FILES log files to process..."

# Find all log files in the current directory and subdirectories
find . -type f -name "training_log.txt" | sort | while read logfile; do
    CURRENT_FILE=$((CURRENT_FILE + 1))
    echo "Processing [$CURRENT_FILE/$TOTAL_FILES]: $logfile..."
    
    if [ "$FIRST_FILE" = true ]; then
        # Process first file without append flag (creates headers)
        python parse_logs.py "$logfile" -o "$OUTPUT_CSV" -r "$TEST_RANGES"
        FIRST_FILE=false
    else
        # Process subsequent files with append flag
        python parse_logs.py "$logfile" -o "$OUTPUT_CSV" -a -r "$TEST_RANGES"
    fi
    
    # Check if the Python script succeeded
    if [ $? -ne 0 ]; then
        echo "Error processing $logfile"
        exit 1
    fi
done

echo ""
echo "All log files processed successfully!"
echo "Results are saved in: $OUTPUT_CSV"

# Check if the output file was created and has content
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "Warning: Output file was not created."
    exit 1
fi

# Generate LaTeX tables if requested
if [ "$GENERATE_LATEX" = true ]; then
    echo ""
    echo "Generating LaTeX tables..."
    python make_latex_tables.py "$OUTPUT_CSV" -r "$TEST_RANGES" -o "$OUTPUT_LATEX"
    
    # Check if LaTeX generation succeeded
    if [ $? -eq 0 ] && [ -f "$OUTPUT_LATEX" ]; then
        echo "LaTeX tables generated successfully: $OUTPUT_LATEX"
    else
        echo "Warning: LaTeX table generation failed or output file not created."
    fi
fi

# Optional: Add a quick summary of the results
echo ""
echo "=== SUMMARY ==="
TOTAL_LINES=$(wc -l < "$OUTPUT_CSV")
DATA_LINES=$((TOTAL_LINES - 1))  # Subtract header line

echo "Total entries processed: $DATA_LINES"
echo "File size: $(du -h "$OUTPUT_CSV" | cut -f1)"

if [ "$DATA_LINES" -gt 0 ]; then
    echo ""
    echo "Unique language/layer combinations:"
    tail -n +2 "$OUTPUT_CSV" | cut -d, -f1,4 | sort | uniq | wc -l
    
    echo ""
    echo "Languages found:"
    tail -n +2 "$OUTPUT_CSV" | cut -d, -f1 | sort | uniq | tr '\n' ' '
    echo ""
    
    echo ""
    echo "Layer counts found:"
    tail -n +2 "$OUTPUT_CSV" | cut -d, -f4 | sort | uniq | tr '\n' ' '
    echo ""
fi

echo ""
echo "Processing complete!"
echo ""
echo "=== OUTPUT FILES ==="
echo "CSV Results: $OUTPUT_CSV ($(du -h "$OUTPUT_CSV" | cut -f1))"
if [ "$GENERATE_LATEX" = true ] && [ -f "$OUTPUT_LATEX" ]; then
    echo "LaTeX Tables: $OUTPUT_LATEX ($(du -h "$OUTPUT_LATEX" | cut -f1))"
fi

