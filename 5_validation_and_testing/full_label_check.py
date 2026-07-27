

import csv
from collections import Counter


def check_complete_label_distribution(csv_filename):
    
    print(f"\nChecking ALL labels in {csv_filename}:")

    with open(csv_filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)

        labels = []
        for row in reader:
            labels.append(int(row[-1]))  

        label_counts = Counter(labels)
        total = len(labels)

        print(f"Total emails: {total}")
        print(f"Ham (0): {label_counts.get(0, 0)}")
        print(f"Spam (1): {label_counts.get(1, 0)}")

        
        if 1 in label_counts:
            first_spam_index = labels.index(1)
            print(f"First spam email at index: {first_spam_index}")

        return label_counts



datasets = ["enron1", "enron2", "enron4"]
splits = ["train", "test"]

print("COMPLETE LABEL DISTRIBUTION CHECK")
print("=" * 50)

for dataset in datasets:
    print(f"\n{dataset.upper()} DATASET:")
    for split in splits:
        csv_file = f"{dataset}_bow_{split}.csv"
        check_complete_label_distribution(csv_file)
