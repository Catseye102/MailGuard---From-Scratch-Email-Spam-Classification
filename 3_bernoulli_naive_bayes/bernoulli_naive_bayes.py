
import os
import csv
import numpy as np
import math
from collections import defaultdict, Counter


class BernoulliNaiveBayes:
    

    def __init__(self, alpha=1.0):
        
        self.alpha = alpha  

        
        self.class_priors = {}  
        self.feature_probs = (
            {}
        )  
        self.classes = []  
        self.vocabulary = []  
        self.vocab_size = 0  

       
        self.class_counts = {}  
        self.word_doc_counts = {}  
        self.total_docs = 0  

        print(f"Initialized Bernoulli Naive Bayes with alpha={alpha}")

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

        
        unique_values = np.unique(features)
        print(f"Feature values in data: {unique_values}")
        if not all(val in [0, 1] for val in unique_values):
            print("⚠️  Warning: Non-binary values detected in Bernoulli data!")

        return features, labels

    def train(self, train_csv_filename):
        
        print(f"\n{'='*60}")
        print("TRAINING BERNOULLI NAIVE BAYES")
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

        
        print(f"\nStep 2: Calculating word document counts per class...")
        self.word_doc_counts = {
            class_label: np.zeros(self.vocab_size) for class_label in self.classes
        }

        for class_label in self.classes:
            
            class_docs = X_train[y_train == class_label]

            
            self.word_doc_counts[class_label] = np.sum(class_docs, axis=0)

            total_docs_in_class = len(class_docs)
            words_appearing_in_class = np.sum(self.word_doc_counts[class_label] > 0)

            print(
                f"Class {class_label}: {total_docs_in_class} docs, {words_appearing_in_class} words appear at least once"
            )

        
        print(
            f"\nStep 3: Calculating feature probabilities with Laplace smoothing (alpha={self.alpha})..."
        )
        self.feature_probs = {}

        for class_label in self.classes:
            self.feature_probs[class_label] = {
                "present": np.zeros(self.vocab_size),  
                "absent": np.zeros(self.vocab_size),  
            }

            
            docs_in_class = self.class_counts[class_label]

            
            for word_idx in range(self.vocab_size):
                
                docs_with_word = self.word_doc_counts[class_label][word_idx]

                
                
                prob_present = (docs_with_word + self.alpha) / (
                    docs_in_class + 2 * self.alpha
                )
                prob_absent = 1.0 - prob_present

                self.feature_probs[class_label]["present"][word_idx] = prob_present
                self.feature_probs[class_label]["absent"][word_idx] = prob_absent

            
            avg_prob_present = np.mean(self.feature_probs[class_label]["present"])
            avg_prob_absent = np.mean(self.feature_probs[class_label]["absent"])
            print(
                f"Class {class_label}: Avg P(word=1|c)={avg_prob_present:.4f}, Avg P(word=0|c)={avg_prob_absent:.4f}"
            )

        print(f"\n[OK] Training completed successfully!")

        
        print(f"\nSample word probabilities:")
        for i in range(min(5, len(self.vocabulary))):
            word = self.vocabulary[i]
            prob_present_ham = self.feature_probs[0]["present"][i]
            prob_present_spam = self.feature_probs[1]["present"][i]
            print(
                f"  '{word}': P(present|ham)={prob_present_ham:.6f}, P(present|spam)={prob_present_spam:.6f}"
            )

    def predict_single(self, feature_vector):
        
        log_probabilities = {}

        for class_label in self.classes:
            
            log_prob = math.log(self.class_priors[class_label])

            
            for word_idx in range(self.vocab_size):
                word_present = feature_vector[word_idx]  # 0 or 1

                if word_present == 1:
                    
                    log_prob += math.log(
                        self.feature_probs[class_label]["present"][word_idx]
                    )
                else:
                    
                    log_prob += math.log(
                        self.feature_probs[class_label]["absent"][word_idx]
                    )

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

        
        print(f"\n EVALUATION RESULTS:")
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


def test_bernoulli_nb_single_dataset(dataset_name):
    
    print(f"\n{'='*80}")
    print(f"TESTING BERNOULLI NAIVE BAYES ON {dataset_name.upper()}")
    print(f"{'='*80}")

    
    nb = BernoulliNaiveBayes(alpha=1.0)

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    datasets_dir = os.path.join(project_root, "generated_datasets")

    
    train_file = os.path.join(datasets_dir, f"{dataset_name}_bernoulli_train.csv")
    nb.train(train_file)

    
    test_file = os.path.join(datasets_dir, f"{dataset_name}_bernoulli_test.csv")
    metrics = nb.evaluate(test_file)

    return metrics


def test_all_datasets():
    
    datasets = ["enron1", "enron2", "enron4"]
    all_results = {}

    print("BERNOULLI NAIVE BAYES - COMPREHENSIVE EVALUATION")
    print("=" * 80)

    for dataset in datasets:
        try:
            results = test_bernoulli_nb_single_dataset(dataset)
            all_results[dataset] = results
        except Exception as e:
            print(f"[ERROR] Error testing {dataset}: {e}")
            all_results[dataset] = {"error": str(e)}

    
    print(f"\n{'='*80}")
    print("BERNOULLI NAIVE BAYES - SUMMARY REPORT")
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
    # Test on all datasets
    results = test_all_datasets()
