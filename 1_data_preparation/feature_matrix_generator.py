

import os
import pickle
import csv
import numpy as np
from collections import Counter
import sys


sys.path.insert(0, os.path.dirname(__file__))
from vocabulary_builder import get_dataset_paths, load_vocabulary


class FeatureMatrixGenerator:
    

    def __init__(self, dataset_name):
        
        self.dataset_name = dataset_name
        self.vocab_data = load_vocabulary(dataset_name)

        if self.vocab_data is None:
            raise ValueError(f"Could not load vocabulary for {dataset_name}")

        self.vocabulary = self.vocab_data["vocabulary"]
        self.vocab_size = self.vocab_data["vocab_size"]
        self.preprocessor = self.vocab_data["preprocessor"]
        self.paths = self.vocab_data["paths"]

        print(f"Loaded vocabulary for {dataset_name}: {self.vocab_size} words")

    def email_to_bow_vector(self, email_tokens):
        
        
        token_counts = Counter(email_tokens)

        
        feature_vector = []
        for word in sorted(self.vocabulary.keys()):
            count = token_counts.get(word, 0)
            feature_vector.append(count)

        return feature_vector

    def email_to_bernoulli_vector(self, email_tokens):
        
        token_set = set(email_tokens)

        
        feature_vector = []
        for word in sorted(self.vocabulary.keys()):
            present = 1 if word in token_set else 0
            feature_vector.append(present)

        return feature_vector

    def process_email_folder(self, folder_path, label, representation="bow"):
       
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found - {folder_path}")
            return []

        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        feature_matrix = []

        print(
            f"Processing {len(txt_files)} emails from {os.path.basename(folder_path)}..."
        )

        for filename in txt_files:
            filepath = os.path.join(folder_path, filename)
            try:
                
                with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                    email_content = file.read()
                    email_tokens = self.preprocessor.preprocess_email(email_content)

                
                if representation == "bow":
                    feature_vector = self.email_to_bow_vector(email_tokens)
                elif representation == "bernoulli":
                    feature_vector = self.email_to_bernoulli_vector(email_tokens)
                else:
                    raise ValueError(f"Unknown representation: {representation}")

                
                feature_row = feature_vector + [label]
                feature_matrix.append(feature_row)

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        return feature_matrix

    def generate_dataset_csv(self, representation="bow", dataset_split="train"):
        
        print(f"\n{'='*60}")
        print(
            f"GENERATING {representation.upper()} {dataset_split.upper()} DATASET FOR {self.dataset_name.upper()}"
        )
        print(f"{'='*60}")

        
        ham_folder = self.paths[f"{dataset_split}_ham"]
        spam_folder = self.paths[f"{dataset_split}_spam"]

        
        ham_matrix = self.process_email_folder(
            ham_folder, label=0, representation=representation
        )

        
        spam_matrix = self.process_email_folder(
            spam_folder, label=1, representation=representation
        )

        
        full_matrix = ham_matrix + spam_matrix

        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        datasets_dir = os.path.join(project_root, "generated_datasets")
        csv_filename = os.path.join(
            datasets_dir, f"{self.dataset_name}_{representation}_{dataset_split}.csv"
        )

        
        vocab_words = sorted(self.vocabulary.keys())
        header = vocab_words + ["label"]

        
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  
            writer.writerows(full_matrix)  

        print(f"\nDataset saved to: {csv_filename}")
        print(f"Shape: {len(full_matrix)} emails × {len(header)} features")
        print(f"Ham emails: {len(ham_matrix)}")
        print(f"Spam emails: {len(spam_matrix)}")

        
        if full_matrix:
            sample_row = full_matrix[0][:-1]  
            non_zero_features = sum(1 for x in sample_row if x > 0)
            print(
                f"Sample email non-zero features: {non_zero_features}/{len(sample_row)}"
            )

        return csv_filename


def generate_all_csv_files():
   
    datasets = ["enron1", "enron2", "enron4"]
    representations = ["bow", "bernoulli"]
    splits = ["train", "test"]

    generated_files = []

    print("EMAIL SPAM CLASSIFICATION - CSV GENERATION")
    print("=" * 60)
    print("Generating 12 CSV files as required by project specifications...")

    for dataset in datasets:
        try:
            generator = FeatureMatrixGenerator(dataset)

            for representation in representations:
                for split in splits:
                    csv_file = generator.generate_dataset_csv(representation, split)
                    generated_files.append(csv_file)

        except Exception as e:
            print(f"Error processing {dataset}: {e}")

    
    print(f"\n{'='*60}")
    print("CSV GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Generated {len(generated_files)} CSV files:")
    for filename in sorted(generated_files):
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f"  {filename} ({file_size:.2f} MB)")


def main():
    
    generate_all_csv_files()


if __name__ == "__main__":
    main()
