#!/bin/bash

# Define the output file
OUTPUT_CSV="combined_results.csv"

# Allow custom output file from command line
if [ "$1" != "" ]; then
    OUTPUT_CSV="$1"
fi

# Remove the output file if it exists (to start fresh)
rm -f "$OUTPUT_CSV"

# Flag to track if this is the first file (for header creation)
FIRST_FILE=true

# Find all log files in the current directory and subdirectories
find . -type f -name "training_log.txt" | sort | while read logfile; do
    echo "Processing $logfile..."
    
    if [ "$FIRST_FILE" = true ]; then
        # Process first file without append flag (creates headers)
        python parse_logs.py "$logfile" -o "$OUTPUT_CSV"
        FIRST_FILE=false
    else
        # Process subsequent files with append flag
        python parse_logs.py "$logfile" -o "$OUTPUT_CSV" -a
    fi
done

echo "All log files processed. Results are in $OUTPUT_CSV"

# Optional: Add a quick summary of the results
echo "Summary of processed files:"
wc -l "$OUTPUT_CSV"
echo "Unique combinations of parameters found:"
tail -n +2 "$OUTPUT_CSV" | cut -d, -f2-4 | sort | uniq | wc -l
