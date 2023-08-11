import sys
import re

def extract_modified_accuracies_from_file(filename, column_names):
    accuracies = {col: [] for col in column_names}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "loocv_module=" in lines[i]:
                modules = re.findall(r"'([^']+)'", lines[i])
                for col in column_names:
                    if col in modules and i + 7 < len(lines):
                        acc_line = lines[i + 7]
                        match = re.search(r': ([\d.]+) \(', acc_line)
                        if match:
                            accuracies[col].append(float(match.group(1)))
    return accuracies

if __name__ == "__main__":
    # Read input arguments
    head_file, wgcna_file, classes_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    
    # Read head file
    with open(head_file, 'r') as f:
        column_names = f.readline().strip().split(',')
    
    # Extract accuracies
    wgcna_accuracies = extract_modified_accuracies_from_file(wgcna_file, column_names)
    classes_accuracies = extract_modified_accuracies_from_file(classes_file, column_names)

    # Calculate average accuracies
    average_accuracies = {}
    for col in column_names:
        all_accuracies = wgcna_accuracies[col] + classes_accuracies[col]
        average_accuracies[col] = sum(all_accuracies) / len(all_accuracies) if all_accuracies else "None"
    
    # Write to output file
    with open(output_file, 'w') as f:
        for col, avg in average_accuracies.items():
            f.write(f"{col}: {avg}\n")
