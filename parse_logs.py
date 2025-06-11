import re
import csv
import sys
import os
from collections import defaultdict

def parse_log_file(file_path):
    """
    Parse the training log file and extract configurations and results.

    Args:
        file_path: Path to the log file

    Returns:
        List of dictionaries with the results for each model configuration
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract language name from file path
    language_match = re.search(r'L(\d+)', file_path)
    language = f"L{language_match.group(1)}" if language_match else "Unknown"

    # Extract layers from file path or training configuration
    layers_match = re.search(r'./(\d+)', file_path)
    layers = layers_match.group(1) if layers_match else None

    # Extract training configuration
    layers_config_match = re.search(r'Layers: (\d+)', content)
    if layers_config_match and not layers:
        layers = layers_config_match.group(1)

    # Check if a model reached 100% accuracy
    success_match = re.search(r'Model (.*?) reached 100% validation accuracy\s*heads: (\d+), dim: (\d+), lr: ([0-9e.-]+)', content)

    # Check if all models failed
    failure_match = re.search(r'All models failed to reach 100% validation accuracy\. Saving best model on validation set\.\nBest model - heads: (\d+), dim: (\d+), lr: ([0-9e.-]+)', content)


    # Check for results
    accuracy_match = re.search(
        r'Accuracy on 100_to_150: ([0-9.]+)%\s*'
        r'Accuracy on 150_to_200: ([0-9.]+)%\s*'
        r'Accuracy on 200_to_250: ([0-9.]+)%\s*'
        r'Accuracy on 250_to_300: ([0-9.]+)%', content)

    # Initialize result with common fields
    result = {
        'language': language,
        'layers': layers,
        'reached_100_percent': False,
        'acc_100_150': 1.0,
        'acc_150_200': 1.0,
        'acc_200_250': 1.0,
        'acc_250_300': 1.0
    }

    if success_match:
        # Extract values from the success message
        model_name = success_match.group(1)
        result['heads'] = int(success_match.group(2))
        result['dim'] = int(success_match.group(3))
        result['lr'] = success_match.group(4)
        result['reached_100_percent'] = True

        # Find the corresponding "Testing with" section for this successful model
        section_pattern = f"Testing with dim: {result['dim']}, heads: {result['heads']}, lr: {result['lr']}"

    elif failure_match:
        print("hi")
        # Extract values from the failure message
        result['heads'] = int(failure_match.group(1))
        result['dim'] = int(failure_match.group(2))
        result['lr'] = failure_match.group(3)
        result['reached_100_percent'] = False

    else:
        # If no success or failure message, return empty result
        print(f"No valid training results found in {file_path}")
        return {}

    if accuracy_match:
        # Extract accuracies from the match
        result['acc_100_150'] = float(accuracy_match.group(1)) 
        result['acc_150_200'] = float(accuracy_match.group(2)) 
        result['acc_200_250'] = float(accuracy_match.group(3)) 
        result['acc_250_300'] = float(accuracy_match.group(4)) 
    return result


def write_to_csv(results, output_file, append=False):
    """
    Write the parsed results to a CSV file.

    Args:
        results: List of dictionaries containing the extracted data
        output_file: Path to the output CSV file
        append: If True, append to existing file, otherwise create new file
    """
    # Define all possible fields that might be in results
    all_possible_fields = [
        'language', 'dim', 'heads', 'layers', 'lr',
        'reached_100_percent',
        'acc_100_150', 'acc_150_200', 'acc_200_250', 'acc_250_300'
    ]

    # Ensure all results have all fields (with default values if missing)
    for result in results:
        for field in all_possible_fields:
            if field not in result:
                result[field] = 0  # Default value for missing fields

    mode = 'a' if append and os.path.exists(output_file) else 'w'

    # If appending and file exists, read existing header to ensure compatibility
    fieldnames = all_possible_fields
    if mode == 'a' and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            existing_header = next(reader, None)
            if existing_header:
                fieldnames = existing_header
                # Ensure all results only have fields from existing header
                for result in results:
                    keys_to_remove = [k for k in result.keys() if k not in fieldnames]
                    for k in keys_to_remove:
                        del result[k]

    write_header = mode == 'w' or not os.path.exists(output_file) or os.path.getsize(output_file) == 0

    with open(output_file, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"Results {'appended to' if append else 'written to'} {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_logs.py <input_log_file> [-o output_csv_file] [-a]")
        print("  -o: specify output CSV file (default: training_results.csv)")
        print("  -a: append to existing CSV file (default: overwrite)")
        sys.exit(1)

    # Default values
    input_file = None
    output_file = "training_results.csv"
    append_mode = False

    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i+1 < len(sys.argv):
            output_file = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '-a':
            append_mode = True
            i += 1
        else:
            input_file = sys.argv[i]
            i += 1

    if input_file is None:
        print("Error: No input file specified.")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Processing file: {input_file}")
    results = [parse_log_file(input_file)]
    print(results)

    if not results:
        print("Warning: No results were extracted from the log file.")
    else:
        write_to_csv(results, output_file, append=append_mode)
        print(f"Extracted {len(results)} configuration(s) and wrote results to {output_file}")

if __name__ == "__main__":
    main()
