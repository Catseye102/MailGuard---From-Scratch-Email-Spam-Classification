

import os
import pickle
import sys


sys.path.insert(0, os.path.dirname(__file__))
from text_preprocessor import EmailPreprocessor


def get_dataset_paths(base_path, dataset_name):
    
    if dataset_name == "enron1":
        return {
            "train_ham": os.path.join(
                base_path, f"{dataset_name}_train", dataset_name, "train", "ham"
            ),
            "train_spam": os.path.join(
                base_path, f"{dataset_name}_train", dataset_name, "train", "spam"
            ),
            "test_ham": os.path.join(
                base_path, f"{dataset_name}_test", dataset_name, "test", "ham"
            ),
            "test_spam": os.path.join(
                base_path, f"{dataset_name}_test", dataset_name, "test", "spam"
            ),
        }
    elif dataset_name == "enron2":
        return {
            "train_ham": os.path.join(
                base_path, f"{dataset_name}_train", "train", "ham"
            ),
            "train_spam": os.path.join(
                base_path, f"{dataset_name}_train", "train", "spam"
            ),
            "test_ham": os.path.join(base_path, f"{dataset_name}_test", "test", "ham"),
            "test_spam": os.path.join(
                base_path, f"{dataset_name}_test", "test", "spam"
            ),
        }
    elif dataset_name == "enron4":
        return {
            "train_ham": os.path.join(
                base_path, f"{dataset_name}_train", dataset_name, "train", "ham"
            ),
            "train_spam": os.path.join(
                base_path, f"{dataset_name}_train", dataset_name, "train", "spam"
            ),
            "test_ham": os.path.join(
                base_path, f"{dataset_name}_test", dataset_name, "test", "ham"
            ),
            "test_spam": os.path.join(
                base_path, f"{dataset_name}_test", dataset_name, "test", "spam"
            ),
        }


def build_all_vocabularies():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_path = os.path.join(project_root, "dataset")

    datasets = ["enron1", "enron2", "enron4"]
    vocabularies = {}

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"BUILDING VOCABULARY FOR {dataset_name.upper()}")
        print(f"{'='*60}")

        
        paths = get_dataset_paths(base_path, dataset_name)

        
        print("Checking paths:")
        for key, path in paths.items():
            exists = os.path.exists(path)
            print(f"  {key}: {exists} - {path}")

        
        preprocessor = EmailPreprocessor()
        vocabulary = preprocessor.build_vocabulary(paths, min_frequency=2)

        
        vocabularies[dataset_name] = {
            "vocabulary": vocabulary,
            "vocab_size": len(vocabulary),
            "preprocessor": preprocessor,
            "paths": paths,
        }

        
        vocab_filename = f"{dataset_name}_vocabulary.pkl"
        with open(vocab_filename, "wb") as f:
            pickle.dump(vocabularies[dataset_name], f)

        print(f"Vocabulary saved to {vocab_filename}")

    return vocabularies


def load_vocabulary(dataset_name):
    
    vocab_filename = f"{dataset_name}_vocabulary.pkl"
    if os.path.exists(vocab_filename):
        with open(vocab_filename, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Vocabulary file {vocab_filename} not found!")
        return None


def main():
    
    print("EMAIL SPAM CLASSIFICATION - VOCABULARY BUILDING")
    print("=" * 60)

    
    vocabularies = build_all_vocabularies()

    
    print(f"\n{'='*60}")
    print("VOCABULARY BUILDING SUMMARY")
    print(f"{'='*60}")

    for dataset_name, vocab_data in vocabularies.items():
        vocab_size = vocab_data["vocab_size"]
        sample_words = list(vocab_data["vocabulary"].keys())[:15]

        print(f"\n{dataset_name.upper()}:")
        print(f"  Vocabulary size: {vocab_size} words")
        print(f"  Sample words: {sample_words}")
        print(f"  Saved to: {dataset_name}_vocabulary.pkl")


if __name__ == "__main__":
    main()
