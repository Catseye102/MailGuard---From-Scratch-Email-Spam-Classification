

import os
import csv
import numpy as np
import math
from collections import Counter
import time


class LogisticRegression:
    

    def __init__(
        self, learning_rate=0.01, max_iterations=500, lambda_reg=1.0, verbose=True
    ):
        
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.lambda_reg = lambda_reg
        self.verbose = verbose

        
        self.weights = None  
        self.num_features = 0

        
        self.loss_history = []

        if self.verbose:
            print(f"Initialized Logistic Regression:")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Lambda (L2 reg): {lambda_reg}")

    def sigmoid(self, z):
        
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def load_data(self, csv_filename):
        
        if self.verbose:
            print(f"Loading data from {csv_filename}...")

        features = []
        labels = []

        with open(csv_filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)

            
            header = next(reader)

            
            for row in reader:
                
                feature_vector = [float(x) for x in row[:-1]]
                label = int(row[-1])

                features.append(feature_vector)
                labels.append(label)

        features = np.array(features)
        labels = np.array(labels)

        if self.verbose:
            print(f"Data loaded: {len(features)} samples, {features.shape[1]} features")
            print(f"Label distribution: {Counter(labels)}")

        return features, labels

    def add_bias_term(self, X):
        
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def compute_log_likelihood(self, X, y, weights):
        
        z = np.dot(X, weights)
        predictions = self.sigmoid(z)

        
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

        
        log_likelihood = np.sum(
            y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
        )

       
        regularization = -(self.lambda_reg / 2.0) * np.sum(weights[1:] ** 2)

        return log_likelihood + regularization

    def train(self, train_csv_filename, validation_split=0.0):
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("TRAINING LOGISTIC REGRESSION")
            print(f"{'='*60}")

        
        X_train, y_train = self.load_data(train_csv_filename)

        
        if validation_split > 0:
            n_samples = len(X_train)
            n_val = int(n_samples * validation_split)
            n_train = n_samples - n_val

            
            indices = np.random.permutation(n_samples)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            X_train = X_train[train_idx]
            y_train = y_train[train_idx]

            if self.verbose:
                print(f"Split into {n_train} training and {n_val} validation samples")

       
        X_train_bias = self.add_bias_term(X_train)
        self.num_features = X_train_bias.shape[1]

        
        np.random.seed(42)
        self.weights = np.random.randn(self.num_features) * 0.01

        if self.verbose:
            print(f"\nInitialized {self.num_features} weights (including bias)")
            print(f"Starting gradient ascent optimization...")

        
        start_time = time.time()

        for iteration in range(self.max_iterations):
           
            z = np.dot(X_train_bias, self.weights)
            predictions = self.sigmoid(z)

            
            errors = y_train - predictions

            
            gradients = np.zeros(self.num_features)

            for j in range(self.num_features):
                
                gradient = np.sum(X_train_bias[:, j] * errors)

                
                if j > 0:
                    gradient -= self.lambda_reg * self.weights[j]

                gradients[j] = gradient

            
            self.weights += self.learning_rate * gradients

            
            if iteration % 50 == 0 or iteration == self.max_iterations - 1:
                loss = self.compute_log_likelihood(X_train_bias, y_train, self.weights)
                self.loss_history.append(loss)

                if self.verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"Iteration {iteration:3d}: Log-likelihood = {loss:.4f} ({elapsed:.2f}s)"
                    )

        elapsed_total = time.time() - start_time

        if self.verbose:
            print(f"\n[OK] Training completed in {elapsed_total:.2f}s")
            print(f"Final log-likelihood: {self.loss_history[-1]:.4f}")

            
            print(f"\nWeight statistics:")
            print(f"  Mean: {np.mean(self.weights):.6f}")
            print(f"  Std: {np.std(self.weights):.6f}")
            print(f"  Min: {np.min(self.weights):.6f}")
            print(f"  Max: {np.max(self.weights):.6f}")

    def predict_proba(self, X):
        
        
        X_bias = self.add_bias_term(X)

        
        z = np.dot(X_bias, self.weights)
        probabilities = self.sigmoid(z)

        return probabilities

    def predict(self, X, threshold=0.5):
        
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, test_csv_filename):
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"EVALUATING ON TEST DATA: {test_csv_filename}")
            print(f"{'='*60}")

        
        X_test, y_test = self.load_data(test_csv_filename)

        
        y_pred = self.predict(X_test)

        
        metrics = self.calculate_metrics(y_test, y_pred)

        
        if self.verbose:
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


def hyperparameter_tuning(dataset_name, representation="bow", lambda_values=None):
    
    if lambda_values is None:
        lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING: {dataset_name.upper()} - {representation.upper()}")
    print(f"{'='*80}")
    print(f"Testing lambda values: {lambda_values}")

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    datasets_dir = os.path.join(project_root, "generated_datasets")
    train_file = os.path.join(
        datasets_dir, f"{dataset_name}_{representation}_train.csv"
    )

    best_lambda = None
    best_accuracy = 0
    results = []

    for lambda_val in lambda_values:
        print(f"\n--- Testing lambda = {lambda_val} ---")

        
        lr = LogisticRegression(
            learning_rate=0.01, max_iterations=500, lambda_reg=lambda_val, verbose=False
        )

        
        X_full, y_full = lr.load_data(train_file)

        
        n_samples = len(X_full)
        n_train = int(0.7 * n_samples)

        
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train = X_full[train_idx]
        y_train = y_full[train_idx]
        X_val = X_full[val_idx]
        y_val = y_full[val_idx]

       
        X_train_bias = lr.add_bias_term(X_train)
        lr.num_features = X_train_bias.shape[1]
        lr.weights = np.random.randn(lr.num_features) * 0.01

        
        for iteration in range(lr.max_iterations):
            z = np.dot(X_train_bias, lr.weights)
            predictions = lr.sigmoid(z)
            errors = y_train - predictions

            gradients = np.zeros(lr.num_features)
            for j in range(lr.num_features):
                gradient = np.sum(X_train_bias[:, j] * errors)
                if j > 0:
                    gradient -= lr.lambda_reg * lr.weights[j]
                gradients[j] = gradient

            lr.weights += lr.learning_rate * gradients

        
        y_val_pred = lr.predict(X_val)
        val_metrics = lr.calculate_metrics(y_val, y_val_pred)

        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1-Score: {val_metrics['f1_score']:.4f}")

        results.append(
            {
                "lambda": lambda_val,
                "accuracy": val_metrics["accuracy"],
                "f1_score": val_metrics["f1_score"],
            }
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_lambda = lambda_val

    print(f"\n{'='*60}")
    print(f"BEST HYPERPARAMETERS:")
    print(f"{'='*60}")
    print(f"Best lambda: {best_lambda}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

    return {
        "best_lambda": best_lambda,
        "best_accuracy": best_accuracy,
        "all_results": results,
    }


def test_logistic_regression_single_dataset(
    dataset_name, representation="bow", lambda_reg=1.0
):
    
    print(f"\n{'='*80}")
    print(
        f"TESTING LOGISTIC REGRESSION: {dataset_name.upper()} - {representation.upper()}"
    )
    print(f"{'='*80}")

    
    lr = LogisticRegression(
        learning_rate=0.01, max_iterations=500, lambda_reg=lambda_reg, verbose=True
    )

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    datasets_dir = os.path.join(project_root, "generated_datasets")

    
    train_file = os.path.join(
        datasets_dir, f"{dataset_name}_{representation}_train.csv"
    )
    lr.train(train_file)

    
    test_file = os.path.join(datasets_dir, f"{dataset_name}_{representation}_test.csv")
    metrics = lr.evaluate(test_file)

    return metrics


def comprehensive_logistic_regression_evaluation():
    
    datasets = ["enron1", "enron2", "enron4"]
    representations = ["bow", "bernoulli"]

    print("LOGISTIC REGRESSION - COMPREHENSIVE EVALUATION")
    print("=" * 80)

    all_results = {}

    for dataset in datasets:
        all_results[dataset] = {}

        for representation in representations:
            print(f"\n{'#'*80}")
            print(f"# PROCESSING: {dataset.upper()} - {representation.upper()}")
            print(f"{'#'*80}")

            try:
                
                tuning_results = hyperparameter_tuning(dataset, representation)
                best_lambda = tuning_results["best_lambda"]

                
                metrics = test_logistic_regression_single_dataset(
                    dataset, representation, lambda_reg=best_lambda
                )

                all_results[dataset][representation] = {
                    "best_lambda": best_lambda,
                    "metrics": metrics,
                }

            except Exception as e:
                print(f"[ERROR] Error: {e}")
                import traceback

                traceback.print_exc()
                all_results[dataset][representation] = {"error": str(e)}

   
    print(f"\n{'='*80}")
    print("LOGISTIC REGRESSION - SUMMARY REPORT")
    print(f"{'='*80}")

    print(
        f"\n{'Dataset':<10} {'Repr.':<10} {'Lambda':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}"
    )
    print("-" * 90)

    for dataset in datasets:
        for representation in representations:
            result = all_results[dataset][representation]
            if "error" not in result:
                metrics = result["metrics"]
                lambda_val = result["best_lambda"]
                print(
                    f"{dataset:<10} {representation:<10} {lambda_val:<10} "
                    f"{metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                    f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}"
                )
            else:
                print(f"{dataset:<10} {representation:<10} ERROR")

    return all_results


if __name__ == "__main__":
    
    results = comprehensive_logistic_regression_evaluation()
