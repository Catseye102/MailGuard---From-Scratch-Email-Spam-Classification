

import csv


def check_labels_in_detail(csv_filename, num_rows_to_check=20):
    """Check labels in detail"""
    print(f"\nChecking labels in {csv_filename}:")

    with open(csv_filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)

        print(f"Header ends with: {header[-5:]}")  
        print(f"Label column index: {len(header)-1}")

        labels = []
        for i, row in enumerate(reader):
            if i >= num_rows_to_check:
                break
            label = row[-1]
            labels.append(label)
            if i < 5:  
                print(f"Row {i}: last 3 values = {row[-3:]}")

        print(f"First {len(labels)} labels: {labels}")
        unique_labels = set(labels)
        print(f"Unique labels found: {unique_labels}")



files_to_check = ["enron1_bow_train.csv", "enron1_bow_test.csv", "enron4_bow_train.csv"]

for filename in files_to_check:
    check_labels_in_detail(filename)
