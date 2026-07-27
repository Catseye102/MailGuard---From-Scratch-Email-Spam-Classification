

import csv
import os
import numpy as np
from collections import Counter
import pickle


def validate_csv_structure(csv_filename):
    
    print(f"\n--- Validating {csv_filename} ---")

    if not os.path.exists(csv_filename):
        return {"error": f"File {csv_filename} does not exist"}

    try:
        with open(csv_filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)

            
            header = next(reader)

            
            data_rows = list(reader)

        
        results = {
            "filename": csv_filename,
            "header_length": len(header),
            "num_rows": len(data_rows),
            "last_column": header[-1],
            "sample_row_length": len(data_rows[0]) if data_rows else 0,
            "file_size_mb": os.path.getsize(csv_filename) / (1024 * 1024),
        }

        
        results["correct_label_column"] = header[-1] == "label"

        
        row_lengths = [len(row) for row in data_rows[:10]]  
        results["consistent_row_lengths"] = all(
            length == len(header) for length in row_lengths
        )

        
        if data_rows:
            labels = [int(row[-1]) for row in data_rows[:100]]  # First 100 labels
            results["label_distribution"] = Counter(labels)

            
            first_row_features = [int(x) for x in data_rows[0][:-1]]
            results["sample_feature_stats"] = {
                "min_value": min(first_row_features),
                "max_value": max(first_row_features),
                "non_zero_count": sum(1 for x in first_row_features if x > 0),
                "zero_count": sum(1 for x in first_row_features if x == 0),
            }

        return results

    except Exception as e:
        return {"error": str(e)}


def validate_bow_vs_bernoulli(bow_file, bernoulli_file):
    
    print(f"\n--- Comparing BoW vs Bernoulli: {bow_file} vs {bernoulli_file} ---")

    if not (os.path.exists(bow_file) and os.path.exists(bernoulli_file)):
        print("One or both files missing")
        return

    try:
       
        with open(bow_file, "r", encoding="utf-8") as f:
            bow_reader = csv.reader(f)
            bow_header = next(bow_reader)
            bow_first_row = next(bow_reader)

        with open(bernoulli_file, "r", encoding="utf-8") as f:
            bern_reader = csv.reader(f)
            bern_header = next(bern_reader)
            bern_first_row = next(bern_reader)

        
        headers_match = bow_header == bern_header
        print(f"Headers match: {headers_match}")

        
        bow_features = [int(x) for x in bow_first_row[:-1]]
        bern_features = [int(x) for x in bern_first_row[:-1]]

        
        bow_max = max(bow_features)
        bow_nonzero = sum(1 for x in bow_features if x > 0)
        bow_has_counts = any(x > 1 for x in bow_features)

        
        bern_max = max(bern_features)
        bern_nonzero = sum(1 for x in bern_features if x > 0)
        bern_only_binary = all(x in [0, 1] for x in bern_features)

        print(
            f"BoW - Max value: {bow_max}, Non-zero features: {bow_nonzero}, Has counts >1: {bow_has_counts}"
        )
        print(
            f"Bernoulli - Max value: {bern_max}, Non-zero features: {bern_nonzero}, Only 0/1 values: {bern_only_binary}"
        )

       
        bow_nonzero_positions = set(i for i, x in enumerate(bow_features) if x > 0)
        bern_nonzero_positions = set(i for i, x in enumerate(bern_features) if x > 0)
        same_positions = bow_nonzero_positions == bern_nonzero_positions

        print(f"Non-zero positions match: {same_positions}")

        
        labels_match = bow_first_row[-1] == bern_first_row[-1]
        print(f"Labels match: {labels_match}")

        return {
            "headers_match": headers_match,
            "bow_has_counts": bow_has_counts,
            "bernoulli_only_binary": bern_only_binary,
            "same_nonzero_positions": same_positions,
            "labels_match": labels_match,
        }

    except Exception as e:
        print(f"Error comparing files: {e}")
        return {"error": str(e)}


def validate_vocabulary_consistency(dataset_name):
    
    print(f"\n--- Validating vocabulary consistency for {dataset_name} ---")

    
    vocab_file = f"{dataset_name}_vocabulary.pkl"
    if not os.path.exists(vocab_file):
        print(f"Vocabulary file {vocab_file} not found")
        return

    with open(vocab_file, "rb") as f:
        vocab_data = pickle.load(f)

    expected_vocab = sorted(vocab_data["vocabulary"].keys())
    expected_vocab_size = len(expected_vocab)

    
    csv_file = f"{dataset_name}_bow_train.csv"
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        csv_vocab = header[:-1]  

        vocab_matches = csv_vocab == expected_vocab
        csv_vocab_size = len(csv_vocab)

        print(f"Expected vocabulary size: {expected_vocab_size}")
        print(f"CSV vocabulary size: {csv_vocab_size}")
        print(f"Vocabulary matches: {vocab_matches}")

        if not vocab_matches:
            print("First 10 expected words:", expected_vocab[:10])
            print("First 10 CSV words:", csv_vocab[:10])

        return {
            "expected_size": expected_vocab_size,
            "csv_size": csv_vocab_size,
            "vocabulary_matches": vocab_matches,
        }


def comprehensive_validation():
    
    print("COMPREHENSIVE CSV VALIDATION")
    print("=" * 60)

    datasets = ["enron1", "enron2", "enron4"]
    representations = ["bow", "bernoulli"]
    splits = ["train", "test"]

    validation_results = {}

    
    print("\n1. BASIC STRUCTURE VALIDATION")
    print("-" * 40)

    for dataset in datasets:
        for representation in representations:
            for split in splits:
                csv_file = f"{dataset}_{representation}_{split}.csv"
                results = validate_csv_structure(csv_file)
                validation_results[csv_file] = results

                if "error" in results:
                    print(f"[ERROR] {csv_file}: {results['error']}")
                else:
                    print(
                        f"[OK] {csv_file}: {results['num_rows']} rows, {results['header_length']} cols, "
                        f"{results['file_size_mb']:.2f}MB"
                    )

   
    print("\n2. BOW VS BERNOULLI VALIDATION")
    print("-" * 40)

    for dataset in datasets:
        for split in splits:
            bow_file = f"{dataset}_bow_{split}.csv"
            bern_file = f"{dataset}_bernoulli_{split}.csv"
            validate_bow_vs_bernoulli(bow_file, bern_file)

    
    print("\n3. VOCABULARY CONSISTENCY VALIDATION")
    print("-" * 40)

    for dataset in datasets:
        validate_vocabulary_consistency(dataset)

    
    print("\n4. LABEL DISTRIBUTION ANALYSIS")
    print("-" * 40)

    for dataset in datasets:
        for split in splits:
            csv_file = f"{dataset}_bow_{split}.csv"  
            if (
                csv_file in validation_results
                and "label_distribution" in validation_results[csv_file]
            ):
                labels = validation_results[csv_file]["label_distribution"]
                total = sum(labels.values())
                ham_count = labels.get(0, 0)
                spam_count = labels.get(1, 0)
                print(
                    f"{dataset} {split}: {ham_count} ham ({ham_count/total*100:.1f}%), "
                    f"{spam_count} spam ({spam_count/total*100:.1f}%)"
                )

    
    print("\n5. VALIDATION SUMMARY")
    print("-" * 40)

    total_files = len([f for f in validation_results.keys() if not f.endswith(".pkl")])
    successful_files = len(
        [
            f
            for f, r in validation_results.items()
            if not f.endswith(".pkl") and "error" not in r
        ]
    )

    print(f"Total CSV files: {total_files}")
    print(f"Successfully validated: {successful_files}")
    print(f"Validation success rate: {successful_files/total_files*100:.1f}%")

    return validation_results


def main():
    """Main validation function"""
    results = comprehensive_validation()

   
    print(f"\n{'='*60}")
    print("PHASE 2 READINESS CHECK")
    print(f"{'='*60}")

    critical_issues = []

    
    for filename, result in results.items():
        if "error" in result:
            critical_issues.append(f"File error: {filename}")

    if not critical_issues:
        print("[OK] ALL CSV FILES VALIDATED SUCCESSFULLY!")
        print("[OK] Ready to proceed to Phase 2: Algorithm Implementation")
    else:
        print("[ERROR] CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            print(f"   - {issue}")


if __name__ == "__main__":
    main()
