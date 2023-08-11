import sys
import re
import csv

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
    head_file, wgcna_file, classes_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    
    with open(head_file, 'r') as f:
        column_names = f.readline().strip().split(',')
    
    wgcna_accuracies = extract_modified_accuracies_from_file(wgcna_file, column_names)
    classes_accuracies = extract_modified_accuracies_from_file(classes_file, column_names)

    results = []
    for col in column_names:
        wgcna_avg = sum(wgcna_accuracies[col]) / len(wgcna_accuracies[col]) if wgcna_accuracies[col] else None
        classes_avg = sum(classes_accuracies[col]) / len(classes_accuracies[col]) if classes_accuracies[col] else None
        overall_avg = None
        inverse_avg = None
        if wgcna_avg is not None and classes_avg is not None:
            overall_avg = (wgcna_avg + classes_avg) / 2
            inverse_avg = 1 - overall_avg
        results.append([wgcna_avg, classes_avg, overall_avg, inverse_avg])
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['WGCNA Accuracy', 'Classes Accuracy', 'Average Accuracy', '1 - Average Accuracy'])
        for col, row in zip(column_names, results):
            writer.writerow([col] + row)
