from typing import Tuple
import numpy as np
from scipy import sparse
import itertools
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from resources import *
from scipy.stats import hmean
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report


# Plot metrics
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


# Evaluate lightFM model
def evaluate_lightfm_model(model, train_data, test_data, k=5):
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


# Run lightFM model
def grid_search_lightfm(interactions, param_grid, k=5, n_jobs=4):
    best_score = -np.inf
    best_params = None
    best_model = None
    evaluations = []

    for params in itertools.product(*param_grid.values()):
        current_params = dict(zip(param_grid.keys(), params))
        print(f"\nTrying parameters:\n{current_params}\n")

        model_lightfm = run_model(
            interactions,
            n_components=current_params["n_components"],
            loss=current_params["loss"],
            epoch=current_params["epoch"],
            n_jobs=n_jobs,
        )
        evaluation = evaluate_lightfm_model(
            model_lightfm, train_sparse, test_sparse, k=k
        )

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
            best_model = model_lightfm

    return best_model, best_params, best_score, evaluations


# set of parameters to try LightFM model
param_grid_lightfm = {
    "n_components": [15, 30, 45],
    "loss": ["warp", "bpr"],
    "epoch": [15, 30, 45],
    "n_jobs": [4],
}


# Step 1: Function to train the KMeans model
def train_kmeans(X: csr_matrix, k: int) -> KMeans:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    return kmeans


# Step 2: Function to visualize the clusters using pyplot
def plot_clusters(X: np.ndarray, kmeans: KMeans, ax: plt.Axes) -> None:
    cmap = plt.cm.get_cmap("viridis", kmeans.n_clusters)
    ax.scatter(
        X[:, 0], X[:, 1], c=kmeans.labels_, cmap=cmap, s=50, edgecolors="k", alpha=0.05
    )
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=200, alpha=0.9)
    ax.set_title(f"KMeans Clustering (k={kmeans.n_clusters})")


# # Step 3: run KMeans with visualization
# def run_kmeans_visualization(X: csr_matrix) -> np.ndarray:
#     elbow_data = []

#     fig, axes = plt.subplots(5, 2, figsize=(12, 18))
#     axes = axes.ravel()

#     for index, k in enumerate(range(2, 11)):
#         kmeans = train_kmeans(X, k)
#         plot_clusters(X, kmeans, axes[index])
#         elbow_data.append(kmeans.inertia_)

#     plt.tight_layout()
#     plt.show()
#     return elbow_data

# #  Step 4: Load & preprocess the data
# def visualize_interactions_data_with_tsne(
#     interactions_csv_path: str,
# ) -> Tuple[csr_matrix, np.ndarray]:
#     # Load user-item interactions matrix from CSV file
#     interactions_df = pd.read_csv(interactions_csv_path)
#     X = csr_matrix(interactions_df.values)

#     # Normalize the data
#     scaler = MaxAbsScaler()
#     X_normalized = scaler.fit_transform(X)

#     # Apply t-SNE for visualization purposes
#     tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
#     X_2d = tsne.fit_transform(X_normalized)

#     return X, X_2d


# Step 3: run KMeans with visualization
def run_kmeans_visualization(X: csr_matrix, X_2d) -> np.ndarray:
    elbow_data = []

    fig, axes = plt.subplots(5, 2, figsize=(12, 18))
    axes = axes.ravel()

    for index, k in enumerate(range(2, 16, 2)):
        kmeans = train_kmeans(X, k)
        plot_clusters(X_2d, kmeans, axes[index])  # Use X_2d only for visualization
        elbow_data.append(kmeans.inertia_)

    plt.tight_layout()
    plt.show()
    return elbow_data


# Step 4: Load & preprocess the data
def visualize_interactions_data_with_tsne(
    interactions_csv_path: str,
) -> Tuple[csr_matrix, np.ndarray]:
    # Load user-item interactions matrix from CSV file
    interactions_df = pd.read_csv(interactions_csv_path)
    X = csr_matrix(interactions_df.values)

    # Normalize the data
    scaler = MaxAbsScaler()
    X_normalized = scaler.fit_transform(X)

    # Apply t-SNE for visualization purposes
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_2d = tsne.fit_transform(X_normalized)

    return X, X_2d


def load_and_preprocess_data(
    interactions_csv_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Load user-item interactions matrix from CSV file
    interactions_df = pd.read_csv(interactions_csv_path)
    X = csr_matrix(interactions_df.values)

    # Define the target variable (y)
    y = np.array(X.sum(axis=1)).flatten()
    y = (y > 0).astype(int)  # Convert to binary values

    # Normalize the data
    scaler = MaxAbsScaler()
    X_normalized = scaler.fit_transform(X)

    # Reduce dimensionality using TruncatedSVD
    svd = TruncatedSVD(n_components=30, random_state=42)
    X_reduced = svd.fit_transform(X_normalized)

    # Apply t-SNE for visualization purposes
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X_normalized)

    return X_reduced, X_2d, y


def visualize_interactions_data_with_tsne_and_svd(
    interactions_csv_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    # Load user-item interactions matrix from CSV file
    interactions_df = pd.read_csv(interactions_csv_path)
    X = csr_matrix(interactions_df.values)

    # Normalize the data
    scaler = MaxAbsScaler()
    X_normalized = scaler.fit_transform(X)

    # Reduce dimensionality using TruncatedSVD
    svd = TruncatedSVD(n_components=30, random_state=42)
    X_reduced = svd.fit_transform(X_normalized)

    # Apply t-SNE for visualization purposes
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_2d = tsne.fit_transform(X_reduced)

    return X_reduced, X_2d


def train_xgboost(X_train, y_train, X_test, y_test, params: Dict) -> Dict:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(params, dtrain)

    y_pred_train = model.predict(dtrain)
    y_pred_test = model.predict(dtest)

    return {
        "model": model,
        "train_predictions": y_pred_train,
        "test_predictions": y_pred_test,
    }


def train_lightgbm(X_train, y_train, X_test, y_test, params: Dict) -> Dict:
    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(params, dtrain, valid_sets=[dtest])

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        "model": model,
        "train_predictions": y_pred_train,
        "test_predictions": y_pred_test,
    }


def train_catboost(X_train, y_train, X_test, y_test, params: Dict) -> Dict:
    model = cb.CatBoostClassifier(**params)

    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        "model": model,
        "train_predictions": y_pred_train,
        "test_predictions": y_pred_test,
    }


def evaluate_classification(y_true, y_pred, threshold=0.5, metrics=["accuracy", "f1"]) -> Dict:
    results = {}

    # Convert the probabilities to binary predictions using the threshold
    y_pred_binary = (y_pred >= threshold).astype(int)

    if "accuracy" in metrics:
        accuracy = accuracy_score(y_true, y_pred_binary)
        results["accuracy"] = accuracy

    if "f1" in metrics:
        f1 = f1_score(y_true, y_pred_binary)
        results["f1"] = f1

    return results


def evaluate_gradient_boosting_models(X: np.ndarray):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate XGBoost Classifier
    xgb_clf = XGBClassifier(objective="binary:logistic", random_state=42)
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    print("XGBoost Classifier Report:")
    print(classification_report(y_test, y_pred_xgb))

    # Train and evaluate LightGBM Classifier
    lgbm_clf = LGBMClassifier(random_state=42)
    lgbm_clf.fit(X_train, y_train)
    y_pred_lgbm = lgbm_clf.predict(X_test)
    print("LightGBM Classifier Report:")
    print(classification_report(y_test, y_pred_lgbm))

    # Train and evaluate CatBoost Classifier
    cat_clf = CatBoostClassifier(random_state=42, verbose=0)
    cat_clf.fit(X_train, y_train)
    y_pred_cat = cat_clf.predict(X_test)
    print("CatBoost Classifier Report:")
    print(classification_report(y_test, y_pred_cat))


def plot_elbow_method(elbow_data: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 16, 2), elbow_data, marker="o", linestyle="--")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.grid()
    plt.show()


def main():
    interactions_csv_path = "interactions.csv"

    # Load and preprocess the data, obtain the 2D representation for visualization
    X_reduced, X_2d = visualize_interactions_data_with_tsne_and_svd(
        interactions_csv_path
    )

    # Train KMeans models with different numbers of clusters and visualize the results
    elbow_data = run_kmeans_visualization(X_reduced)

    # Analyze KMeans models using the elbow method
    plot_elbow_method(elbow_data)

    # Train and evaluate gradient boosting models
    evaluate_gradient_boosting_models(X_reduced)


if __name__ == "__main__":
    main()
