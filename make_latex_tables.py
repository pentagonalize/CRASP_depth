import sys

# Default test ranges (used if none provided via command line)
DEFAULT_TEST_RANGES = [
    (300, 350),
    (350, 400),
    (400, 450),
    (450, 500)
]

def parse_test_ranges(ranges_str):
    """
    Parse test ranges from command line string.
    Expected format: "300,350;350,400;400,450;450,500"
    Returns list of tuples.
    """
    try:
        ranges = []
        for range_pair in ranges_str.split(';'):
            start, end = map(int, range_pair.split(','))
            ranges.append((start, end))
        return ranges
    except (ValueError, IndexError) as e:
        print(f"Error parsing test ranges '{ranges_str}': {e}")
        print("Expected format: 'start1,end1;start2,end2;...'")
        print("Example: '300,350;350,400;400,450;450,500'")
        sys.exit(1)

def get_range_label(start, end):
    """Generate a label for a test range."""
    return f"{start}_{end}"

def get_accuracy_field_name(start, end):
    """Generate field name for accuracy results."""
    return f"acc_{start}_{end}"

def get_latex_title(start, end):
    """Generate LaTeX title for a test range."""
    return f"[{start},{end}]"

def create_latex_table(bin_dict, title):
    latex_str = f"""
\\begin{{center}}
    Accuracy on {title}
    """
    latex_str_end = """
    \\setlength{\\tabcolsep}{0pt}
    \\renewcommand{\\arraystretch}{0}
        \\begin{tabular}[b]{l@{\;\;}*{10}{c}}
        & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\\\[1ex]
    """
    latex_str += latex_str_end
    # bin_dict is a dictionary with languages as keys and a list of accuracies as values
    bin_dict = dict(sorted(bin_dict.items(), key=lambda x: int(x[0][1:])))
    for language, accuracies in bin_dict.items():

        if language.startswith("L"):
            k = int(language[1:])
        latex_str += f"$\\altplus{{{k}}}$ & "

        for i in sorted(accuracies.keys()):
            accuracy = accuracies[i]
            if accuracy is not None:
                latex_str += f"\\cellgradient{{{accuracy}}} & "
            else:
                latex_str += "0.00 & "
        latex_str = latex_str[:-2] + "\\\\\n"
    latex_str += """
    \\end{tabular}%
\\hspace{0.5\\bsize}\\llap{\\raisebox{-0.3\\cellheight}{%
\\begin{tikzpicture}[x=\\cellwidth,y=\\cellheight,baseline=0pt,line width=\\bsize]
\\draw (0,10) |- (1,9) |- (2,8) |- (3,7) |- (4,6) |- (5,5) |- (6,4) |- (7,3) |- (8,2) |- (9,1) |- (10,0);
\\end{tikzpicture}%
}}
\\end{center}
"""
    return latex_str

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_latex_tables.py <csv_file_path> [-r test_ranges] [-o output_file]")
        print("  -r: specify test ranges (format: 'start1,end1;start2,end2;...')")
        print("  -o: specify output LaTeX file (default: latex_tables_combined.tex)")
        print(f"      example: -r '300,350;350,400;400,450;450,500'")
        print(f"Default test ranges: {DEFAULT_TEST_RANGES}")
        sys.exit(1)

    # Default values
    csv_file_path = None
    test_ranges = DEFAULT_TEST_RANGES
    output_file = "latex_tables_combined.tex"

    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-r' and i+1 < len(sys.argv):
            test_ranges = parse_test_ranges(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '-o' and i+1 < len(sys.argv):
            output_file = sys.argv[i+1]
            i += 2
        else:
            csv_file_path = sys.argv[i]
            i += 1

    if csv_file_path is None:
        print("Error: No CSV file specified.")
        sys.exit(1)
    try:
        with open(csv_file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Parse the CSV file manually
    header = lines[0].strip().split(",")
    data = [line.strip().split(",") for line in lines[1:]]

    # Generate bins dictionary from test_ranges
    bins = {}
    for start, end in test_ranges:
        range_label = get_range_label(start, end)
        bins[range_label] = (start, end)

    print(f"Processing CSV file: {csv_file_path}")
    print(f"Using test ranges: {test_ranges}")
    print(f"Processing bins: {list(bins.keys())}")

    grouped_data = {}

    for bin_name, (lower, upper) in bins.items():
        grouped_data[bin_name] = {}
        for row in data:
            row_dict = dict(zip(header, row))
            for col_name in header:
                if col_name.startswith("acc_"):
                    # Extract the range from column name (e.g., "acc_300_350" -> "300_350")
                    col_parts = col_name.split("_")
                    if len(col_parts) >= 3:
                        bin_range = "_".join(col_parts[1:3])  # Join the numeric parts
                        if bin_range == bin_name:
                            language = row_dict["language"]
                            layer = int(row_dict["layers"])
                            accuracy = float(row_dict[col_name]) if row_dict[col_name] else None

                            if language not in grouped_data[bin_name]:
                                grouped_data[bin_name][language] = {}
                            if layer not in grouped_data[bin_name][language]:
                                grouped_data[bin_name][language][layer] = None

                            grouped_data[bin_name][language][layer] = accuracy

    # Convert the grouped data to a list of lists for LaTeX table generation
    data_list = []
    for bin_name, languages in grouped_data.items():
        for language, layers in languages.items():
            row = []
            for layer in range(3, 13):
                if layer in layers:
                    row.append(layers[layer])
                else:
                    row.append(0)  # Default value for missing layers
            data_list.append(row)

    with open(output_file, "w") as f:
        for bin_name, bin_dict in grouped_data.items():
            # Convert bin_name to LaTeX title format
            parts = bin_name.split("_")
            if len(parts) >= 2:
                start, end = parts[0], parts[1]
                latex_title = get_latex_title(start, end)
            else:
                latex_title = f"[{bin_name}]"
            
            latex_table = create_latex_table(bin_dict, latex_title)
            f.write(latex_table)
            # f.write("\n\\newpage\n")
        print(f"All LaTeX tables written to {output_file}")

    print("Done.")
