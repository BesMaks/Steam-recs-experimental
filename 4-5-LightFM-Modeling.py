import numpy as np
from scipy import sparse
import itertools
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
import pandas as pd
from resources import *
from scipy.stats import hmean
import matplotlib.pyplot as plt


def plot_evaluation_metrics(evaluations):
    metrics = list(evaluations[0].keys())
    train_metrics = [m for m in metrics if m.startswith("train")]
    test_metrics = [m for m in metrics if m.startswith("test")]

    for train_metric, test_metric in zip(train_metrics, test_metrics):
        train_values = [evaluation[train_metric] for evaluation in evaluations]
        test_values = [evaluation[test_metric] for evaluation in evaluations]

        plt.figure()
        plt.plot(train_values, label="Train")
        plt.plot(test_values, label="Test")
        plt.xlabel("Parameter Set")
        plt.ylabel("Score")
        plt.title(f"{train_metric[6:].capitalize()} Score")
        plt.legend()
        plt.show()


def evaluate_model(model, train_data, test_data, k=5):
    train_precision = precision_at_k(model, train_data, k=5).mean()
    test_precision = precision_at_k(model, test_data, k=5).mean()

    train_recall = recall_at_k(model, train_data, k=5).mean()
    test_recall = recall_at_k(model, test_data, k=5).mean()

    train_auc = auc_score(model, train_data).mean()
    test_auc = auc_score(model, test_data).mean()

    return {
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_auc": train_auc,
        "test_auc": test_auc,
    }


def grid_search(interactions, param_grid, k=5, n_jobs=4):
    best_score = -np.inf
    best_params = None
    best_model = None
    evaluations = []

    for params in itertools.product(*param_grid.values()):
        current_params = dict(zip(param_grid.keys(), params))
        print(f"\nTrying parameters:\n{current_params}\n")

        model = run_model(
            interactions,
            n_components=current_params["n_components"],
            loss=current_params["loss"],
            epoch=current_params["epoch"],
            n_jobs=n_jobs,
        )
        evaluation = evaluate_model(model, train_sparse, test_sparse, k=k)

        evaluations.append(evaluation)  # Store evaluation results

        print("Evaluation results:")
        for metric, value in evaluation.items():
            print(f"  {metric}: {value:.4f}")
        print()

        test_metrics = [
            evaluation["test_precision"],
            evaluation["test_recall"],
            evaluation["test_auc"],
        ]
        aggregated_score = hmean(
            test_metrics
        )  # Calculate the harmonic mean of test metrics

        if aggregated_score > best_score:
            best_score = aggregated_score
            best_params = current_params
            best_model = model

    return best_model, best_params, best_score, evaluations


param_grid = {
    "n_components": [15, 30, 45],
    "loss": ["warp", "bpr"],
    "epoch": [15, 30, 45],
    "n_jobs": [4],
}


if __name__ == "__main__":
    # Establish number of users in train/test sets
    interactions = pd.read_csv("interactions.csv")
    train_num = round((85 / 100) * len(interactions), 0)
    print(f"We desire {train_num} users in our training set.")

    test_num = len(interactions) - train_num
    print(f"We desire {test_num} users in our test set.")

    # Define train and test sets
    train = interactions[: int(train_num)]
    test = interactions[int(test_num) :]

    # Create sparse matrices for evaluation
    train_sparse = sparse.csr_matrix(train.values)
    # Add X users to Test so that the number of rows in Train match Test
    N = train.shape[0]  # Rows in Train set
    n, m = test.shape  # Rows & columns in Test set
    z = np.zeros([(N - n), m])  # Create the necessary rows of zeros with m columns
    test = np.vstack((test, z))  # Vertically stack Test on top of the blank users
    test_sparse = sparse.csr_matrix(test)  # Convert back to sparse
    print(train_sparse.get_shape())

    # Run the grid search with the modified function
    best_model, best_params, best_score, evaluations = grid_search(train, param_grid)

    print("\nBest parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest aggregated score (harmonic mean) on test set: {best_score:.4f}")

    plot_evaluation_metrics(evaluations)
