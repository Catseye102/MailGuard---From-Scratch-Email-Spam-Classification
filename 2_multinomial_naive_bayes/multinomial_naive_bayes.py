

import os
import csv
import numpy as np
import math
from collections import defaultdict, Counter


class MultinomialNaiveBayes:
    

    def __init__(self, alpha=1.0):
        
        self.alpha = alpha  

        
        self.class_priors = {}  
        self.feature_likelihoods = {}  
        self.classes = []  
        self.vocabulary = []  
        self.vocab_size = 0  

        
        self.class_counts = {}  
        self.word_counts = {}  
        self.total_docs = 0  

        print(f"Initialized Multinomial Naive Bayes with α={alpha}")

    def load_data(self, csv_filename):
        
        print(f"Loading data from {csv_filename}...")

        features = []
        labels = []

        with open(csv_filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)

            
            header = next(reader)
            vocab_from_file = header[:-1]  

            
            if not self.vocabulary:
                self.vocabulary = vocab_from_file
                self.vocab_size = len(self.vocabulary)
                print(f"Vocabulary loaded: {self.vocab_size} words")

            
            for row in reader:
                
                feature_vector = [int(x) for x in row[:-1]]
                label = int(row[-1])

                features.append(feature_vector)
                labels.append(label)

        features = np.array(features)
        labels = np.array(labels)

        print(f"Data loaded: {len(features)} samples, {features.shape[1]} features")
        print(f"Label distribution: {Counter(labels)}")

        return features, labels

    def train(self, train_csv_filename):
        
        print(f"\n{'='*60}")
        print("TRAINING MULTINOMIAL NAIVE BAYES")
        print(f"{'='*60}")

        
        X_train, y_train = self.load_data(train_csv_filename)

        self.total_docs = len(X_train)
        self.classes = sorted(list(set(y_train)))

        print(f"Training on {self.total_docs} documents")
        print(f"Classes: {self.classes}")

        
        print("\nStep 1: Calculating class priors P(c)...")
        self.class_counts = Counter(y_train)

        for class_label in self.classes:
            
            self.class_priors[class_label] = (
                self.class_counts[class_label] / self.total_docs
            )
            print(
                f"P(class={class_label}) = {self.class_counts[class_label]}/{self.total_docs} = {self.class_priors[class_label]:.4f}"
            )

        
        print(f"\nStep 2: Calculating word counts per class...")
        self.word_counts = {
            class_label: np.zeros(self.vocab_size) for class_label in self.classes
        }

        for class_label in self.classes:
            
            class_docs = X_train[y_train == class_label]

            
            self.word_counts[class_label] = np.sum(class_docs, axis=0)

            total_words_in_class = np.sum(self.word_counts[class_label])
            unique_words_in_class = np.sum(self.word_counts[class_label] > 0)

            print(
                f"Class {class_label}: {len(class_docs)} docs, {total_words_in_class} total words, {unique_words_in_class} unique words"
            )

        
        print(
            f"\nStep 3: Calculating feature likelihoods P(w|c) with Laplace smoothing (α={self.alpha})..."
        )
        self.feature_likelihoods = {}

        for class_label in self.classes:
            self.feature_likelihoods[class_label] = np.zeros(self.vocab_size)

            
            total_words_in_class = np.sum(self.word_counts[class_label])

            
            for word_idx in range(self.vocab_size):
                word_count_in_class = self.word_counts[class_label][word_idx]

                
                smoothed_probability = (word_count_in_class + self.alpha) / (
                    total_words_in_class + self.alpha * self.vocab_size
                )

                self.feature_likelihoods[class_label][word_idx] = smoothed_probability

            
            prob_sum = np.sum(self.feature_likelihoods[class_label])
            print(f"Class {class_label}: Feature likelihoods sum = {prob_sum:.6f}")

        print(f"\n✅ Training completed successfully!")

        
        print(f"\nSample word probabilities:")
        for i in range(min(5, len(self.vocabulary))):
            word = self.vocabulary[i]
            prob_ham = self.feature_likelihoods[0][i]
            prob_spam = self.feature_likelihoods[1][i]
            print(f"  '{word}': P(w|ham)={prob_ham:.6f}, P(w|spam)={prob_spam:.6f}")

    def predict_single(self, feature_vector):
        
        log_probabilities = {}

        for class_label in self.classes:
           
            log_prob = math.log(self.class_priors[class_label])

            
            for word_idx in range(self.vocab_size):
                word_count_in_doc = feature_vector[word_idx]
                if (
                    word_count_in_doc > 0
                ):  
                    word_log_likelihood = math.log(
                        self.feature_likelihoods[class_label][word_idx]
                    )
                    log_prob += word_count_in_doc * word_log_likelihood

            log_probabilities[class_label] = log_prob

        
        predicted_class = max(log_probabilities, key=log_probabilities.get)

        return predicted_class

    def predict(self, X_test):
        
        print(f"Predicting on {len(X_test)} test samples...")

        predictions = []
        for i, feature_vector in enumerate(X_test):
            prediction = self.predict_single(feature_vector)
            predictions.append(prediction)

            
            if (i + 1) % 100 == 0 or i == len(X_test) - 1:
                print(f"Processed {i + 1}/{len(X_test)} samples")

        return np.array(predictions)

    def evaluate(self, test_csv_filename):
        
        print(f"\n{'='*60}")
        print(f"EVALUATING ON TEST DATA: {test_csv_filename}")
        print(f"{'='*60}")

        
        X_test, y_test = self.load_data(test_csv_filename)

       
        y_pred = self.predict(X_test)

        
        metrics = self.calculate_metrics(y_test, y_pred)

       
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"{'='*40}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")

        
        print(f"\n📋 DETAILED RESULTS:")
        print(f"True Positives (TP):  {metrics['tp']}")
        print(f"False Positives (FP): {metrics['fp']}")
        print(f"True Negatives (TN):  {metrics['tn']}")
        print(f"False Negatives (FN): {metrics['fn']}")

        return metrics

    def calculate_metrics(self, y_true, y_pred):
        
        tp = np.sum((y_true == 1) & (y_pred == 1))  
        fp = np.sum((y_true == 0) & (y_pred == 1))  
        tn = np.sum((y_true == 0) & (y_pred == 0))  
        fn = np.sum((y_true == 1) & (y_pred == 0))  

        
        accuracy = (tp + tn) / (tp + fp + tn + fn)

        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }


def test_multinomial_nb_single_dataset(dataset_name):
    
    print(f"\n{'='*80}")
    print(f"TESTING MULTINOMIAL NAIVE BAYES ON {dataset_name.upper()}")
    print(f"{'='*80}")

    
    nb = MultinomialNaiveBayes(alpha=1.0)

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    datasets_dir = os.path.join(project_root, "generated_datasets")

    
    train_file = os.path.join(datasets_dir, f"{dataset_name}_bow_train.csv")
    nb.train(train_file)

    
    test_file = os.path.join(datasets_dir, f"{dataset_name}_bow_test.csv")
    metrics = nb.evaluate(test_file)

    return metrics


def test_all_datasets():
    
    datasets = ["enron1", "enron2", "enron4"]
    all_results = {}

    print("MULTINOMIAL NAIVE BAYES - COMPREHENSIVE EVALUATION")
    print("=" * 80)

    for dataset in datasets:
        try:
            results = test_multinomial_nb_single_dataset(dataset)
            all_results[dataset] = results
        except Exception as e:
            print(f"❌ Error testing {dataset}: {e}")
            all_results[dataset] = {"error": str(e)}

    
    print(f"\n{'='*80}")
    print("MULTINOMIAL NAIVE BAYES - SUMMARY REPORT")
    print(f"{'='*80}")

    print(
        f"{'Dataset':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}"
    )
    print("-" * 60)

    for dataset, results in all_results.items():
        if "error" not in results:
            print(
                f"{dataset:<10} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                f"{results['recall']:<10.4f} {results['f1_score']:<10.4f}"
            )
        else:
            print(f"{dataset:<10} ERROR: {results['error']}")

    return all_results


if __name__ == "__main__":
    
    results = test_all_datasets()
