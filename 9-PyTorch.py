import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split


def calculate_metrics(true_data: np.ndarray, pred_data: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(true_data, pred_data)
    mse = mean_squared_error(true_data, pred_data)
    return {"MAE": mae, "MSE": mse}


def print_user_games_and_predictions(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    reduced_matrix: np.ndarray,
    model: nn.Module,
    transformer: TruncatedSVD,
    scaler: MinMaxScaler,
    games_data_file: str,
    n_recommendations: int = 5,
) -> None:
    # Load the games data and create a dictionary to map game ids to titles
    games_data = pd.read_json(games_data_file, dtype={"id": int})
    id_to_title = games_data.set_index("id")["title"].to_dict()

    # Update this line to get the index of the user_id in the matrix
    user_index = user_item_matrix.index.get_loc(user_id)
    user_reduced_vector = reduced_matrix[user_index]

    # Get the model predictions in the reduced space
    with torch.no_grad():
        user_reduced_vector_tensor = torch.tensor(
            user_reduced_vector, dtype=torch.float32
        ).unsqueeze(0)
        predicted_reduced_vector_tensor = model(user_reduced_vector_tensor)
        predicted_reduced_vector = predicted_reduced_vector_tensor.numpy().squeeze()

    # Inverse transformations: denormalize and inverse SVD
    denormalized_predicted_vector = scaler.inverse_transform([predicted_reduced_vector])
    reconstructed_predicted_vector = transformer.inverse_transform(
        denormalized_predicted_vector
    ).squeeze()

    # Get the actual and predicted games for the user
    user_actual_games = user_item_matrix.loc[user_id]
    user_predicted_games = pd.Series(
        reconstructed_predicted_vector, index=user_actual_games.index
    )
    actual_games_sorted = user_actual_games.sort_values(ascending=False)
    predicted_games_sorted = user_predicted_games.sort_values(ascending=False)

    # Filter owned games
    owned_game_ids = set(actual_games_sorted[actual_games_sorted == 1].index)
    owned_games = {
        game_id: id_to_title[game_id]
        for game_id in owned_game_ids
        if game_id in id_to_title
    }

    # Get top recommended games excluding owned games
    recommended_game_ids = [
        game_id
        for game_id in predicted_games_sorted.index
        if game_id not in owned_game_ids
    ][:n_recommendations]
    recommended_games = {
        game_id: id_to_title[game_id]
        for game_id in recommended_game_ids
        if game_id in id_to_title
    }

    # Print the actual and top 5 predicted games for the user
    with open("9-NN-recs.txt", "w") as f:
        f.write(f"User id: {user_id}\n\n")

        f.write("Actual games:\n")
        for game_id in owned_game_ids:
            if game_id in owned_games:
                f.write(f"{owned_games[game_id]} (id: {game_id})\n")
            else:
                f.write(f"Game not found in 'gamesdata' (id: {game_id})\n")

        f.write("\nRecommended games:\n")
        for game_id in recommended_game_ids:
            if game_id in recommended_games:
                f.write(f"{recommended_games[game_id]} (id: {game_id})\n")
            else:
                f.write(f"Game not found in 'gamesdata' (id: {game_id})\n")


def reduce_dimensionality(
    matrix: pd.DataFrame, n_components: int = 50
) -> Tuple[np.ndarray, int]:
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(matrix)
    return reduced_matrix, n_components, svd


class RecommendationNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(RecommendationNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.5),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(0.5),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 4),
            nn.Dropout(0.5),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(0.5),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.5),
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.output_layer(x)
        return x


def save_model(model: torch.nn.Module, file_path: str) -> None:
    torch.save(model.state_dict(), file_path)


def load_model(
    model_class: torch.nn.Module, input_size: int, hidden_size: int, file_path: str
) -> torch.nn.Module:
    model = model_class(input_size, hidden_size)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model


def find_user_with_closest_to_50_games(interaction_matrix):
    # Calculate the sum of games for each user and find the user with the closest number of games to 50
    user_with_closest_to_50 = (
        interaction_matrix.sum(axis=1, skipna=True).sub(50).abs().idxmin()
    )
    # Print the result
    print(
        f"The user with the closest number of games to 50 is {user_with_closest_to_50} with {interaction_matrix.loc[user_with_closest_to_50].sum()} games."
    )


def normalize_data(matrix: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize the input matrix using Min-Max scaling.

    Args:
        matrix (np.ndarray): The input matrix to be normalized.

    Returns:
        Tuple[np.ndarray, MinMaxScaler]: The normalized matrix and the MinMaxScaler object used.
    """
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(matrix)
    return normalized_matrix, scaler


def train_nn(
    model: nn.Module,
    train_data: torch.Tensor,
    eval_data: torch.Tensor,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> Tuple[nn.Module, List[float], Dict[str, List[float]]]:
    model.to(device)
    train_data = train_data.to(device)
    eval_data = eval_data.to(device)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, verbose=True
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_loss_history = []
    eval_loss_history = {"MAE": [], "MSE": []}

    # TensorBoard logging
    log_dir = os.path.join(
        "runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    writer = SummaryWriter(log_dir)

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)
        writer.add_scalar("training loss", epoch_train_loss, epoch)

        # Evaluation
        model.eval()
        with torch.no_grad():
            eval_outputs = model(eval_data)
            eval_outputs_np = eval_outputs.cpu().numpy()
            eval_data_np = eval_data.cpu().numpy()
            metrics = calculate_metrics(eval_data_np, eval_outputs_np)

            for metric_name, metric_value in metrics.items():
                eval_loss_history[metric_name].append(metric_value)
                writer.add_scalar(f"evaluation {metric_name}", metric_value, epoch)

        scheduler.step(epoch_train_loss)

    writer.close()
    return model, train_loss_history, eval_loss_history



def mean_absolute_error_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def root_mean_squared_error_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def average_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    y_true_sorted = np.argsort(y_true)[-k:][::-1]
    y_pred_sorted = np.argsort(y_pred)[-k:][::-1]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(y_pred_sorted):
        if p in y_true_sorted and p not in y_pred_sorted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(y_true_sorted), k)


def evaluate_trained_model(
    model: nn.Module, reduced_matrix: np.ndarray
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(reduced_matrix, dtype=torch.float32))
        y_pred = y_pred.cpu().numpy()

    metrics = {
        "MAE": mean_absolute_error_metric(reduced_matrix, y_pred),
        "RMSE": root_mean_squared_error_metric(reduced_matrix, y_pred),
        "MAP@5": average_precision_at_k(reduced_matrix, y_pred),
    }
    return metrics


def recommend_top_5_items(
    model: nn.Module, reduced_matrix: np.ndarray, user_id: int
) -> List[int]:
    model.to(device)
    model.eval()
    user_vector = torch.tensor(reduced_matrix[user_id], dtype=torch.float32).to(device)
    with torch.no_grad():
        predicted_vector = model(user_vector).cpu().numpy()
    top_5_item_indices = np.argsort(predicted_vector)[-5:][::-1]
    return top_5_item_indices.tolist()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data & reduce dimensionality
    matrix = pd.read_csv("interactions.csv")
    # find_user_with_closest_to_50_games(matrix)
    reduced_matrix, n_components, svd_transformer = reduce_dimensionality(matrix)
    normalized_reduced_matrix, normalizer = normalize_data(reduced_matrix)

    # Define model parameters
    input_size = n_components
    hidden_size = 512
    epochs = 150
    learning_rate = 1e-2
    batch_size = 128

    recommendation_model = RecommendationNN(input_size, hidden_size)
    # Split data
    train_data, eval_data = train_test_split(
        normalized_reduced_matrix, test_size=0.2, random_state=42
    )
    print("train shape:", train_data.shape, "test shape:", eval_data.shape)
    # region Train the model
    # # Convert data to tensors
    # train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    # eval_data_tensor = torch.tensor(eval_data, dtype=torch.float32)

    # # Train the model using the updated train_nn function
    # trained_model, training_history, eval_history = train_nn(
    #     recommendation_model,
    #     train_data_tensor,
    #     eval_data_tensor,
    #     epochs,
    #     learning_rate,
    #     batch_size,
    # )

    # Save the trained model
    # save_model(trained_model, "trained_recommendation_model.pth")
    # endregion

    # Load the trained model
    loaded_model = load_model(
        RecommendationNN, input_size, hidden_size, "trained_recommendation_model.pth"
    )

    games_data = "gamesdata.json"
    user_id = 2
    print_user_games_and_predictions(
        user_id,
        matrix,
        normalized_reduced_matrix,
        loaded_model,
        svd_transformer,
        normalizer,
        games_data,
    )
