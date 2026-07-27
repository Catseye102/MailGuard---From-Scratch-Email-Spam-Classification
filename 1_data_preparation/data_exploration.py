

import os
import glob
from collections import Counter


def count_files_in_dataset(base_path, dataset_name):
    
    print(f"\n=== {dataset_name.upper()} DATASET ===")

    
    if dataset_name == "enron1":
        train_ham_path = os.path.join(
            base_path, f"{dataset_name}_train", dataset_name, "train", "ham"
        )
        train_spam_path = os.path.join(
            base_path, f"{dataset_name}_train", dataset_name, "train", "spam"
        )
        test_ham_path = os.path.join(
            base_path, f"{dataset_name}_test", dataset_name, "test", "ham"
        )
        test_spam_path = os.path.join(
            base_path, f"{dataset_name}_test", dataset_name, "test", "spam"
        )
    elif dataset_name == "enron2":
        train_ham_path = os.path.join(
            base_path, f"{dataset_name}_train", "train", "ham"
        )
        train_spam_path = os.path.join(
            base_path, f"{dataset_name}_train", "train", "spam"
        )
        test_ham_path = os.path.join(base_path, f"{dataset_name}_test", "test", "ham")
        test_spam_path = os.path.join(base_path, f"{dataset_name}_test", "test", "spam")
    elif dataset_name == "enron4":
        train_ham_path = os.path.join(
            base_path, f"{dataset_name}_train", dataset_name, "train", "ham"
        )
        train_spam_path = os.path.join(
            base_path, f"{dataset_name}_train", dataset_name, "train", "spam"
        )
        test_ham_path = os.path.join(
            base_path, f"{dataset_name}_test", dataset_name, "test", "ham"
        )
        test_spam_path = os.path.join(
            base_path, f"{dataset_name}_test", dataset_name, "test", "spam"
        )

    
    paths = {
        "Training Ham": train_ham_path,
        "Training Spam": train_spam_path,
        "Test Ham": test_ham_path,
        "Test Spam": test_spam_path,
    }

    for label, path in paths.items():
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith(".txt")])
            print(f"{label}: {count} emails")
        else:
            print(f"{label}: Path not found - {path}")

    return paths


def read_sample_emails(paths):
    
    print("\n=== SAMPLE EMAILS ===")

    for label, path in paths.items():
        if os.path.exists(path):
            txt_files = [f for f in os.listdir(path) if f.endswith(".txt")]
            if txt_files:
                sample_file = txt_files[0]  
                sample_path = os.path.join(path, sample_file)

                print(f"\n--- Sample {label} ---")
                print(f"File: {sample_file}")
                try:
                    with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()[:300]  
                        print(f"Content preview:\n{content}...")
                except Exception as e:
                    print(f"Error reading file: {e}")


def main():
    
    base_path = r"c:\Users\Ashfaq\Documents\project_1\project 1\dataset"

    print("EMAIL SPAM CLASSIFICATION - DATA EXPLORATION")
    print("=" * 50)

    
    datasets = ["enron1", "enron2", "enron4"]

    for dataset in datasets:
        paths = count_files_in_dataset(base_path, dataset)
        read_sample_emails(paths)
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
