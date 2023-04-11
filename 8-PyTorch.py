import datetime
from typing import List, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import IterableDataset
from scipy.sparse import coo_matrix, save_npz, load_npz


class InteractionMatrixDataset(data.Dataset):
    def __init__(self, interactions: List[Tuple[int, int, int]]):
        # Initialize user-item interactions
        self.interactions = [(user, item) for user, item, label in interactions]
        self.labels = [label for _, _, label in interactions]

    def __getitem__(self, index: int) -> Tuple[int, int]:
        # Get the (user, item) pair based on the index
        return self.interactions[index]

    def get_label(self, index: int) -> int:
        # Get the label based on the index
        return self.labels[index]

    def __len__(self) -> int:
        # Return the number of interactions
        return len(self.interactions)


class InteractionMatrixDataLoader(data.DataLoader):
    def __iter__(self):
        iterator = super().__iter__()
        for batch in iterator:
            user_indices, item_indices = batch
            labels = [self.dataset.get_label(i) for i in range(user_indices.size(0))]
            yield user_indices, item_indices, torch.tensor(labels, dtype=torch.float)


class InteractionMatrixGenerator(IterableDataset):
    def __init__(self, interactions: List[Tuple[int, int, int]], batch_size: int):
        self.interactions = interactions
        self.batch_size = batch_size
        self.num_interactions = len(interactions)

    def __iter__(self):
        indices = np.random.permutation(self.num_interactions)
        for i in range(0, self.num_interactions, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            users, items, labels = [], [], []
            for idx in batch_indices:
                user, item, label = self.interactions[idx]
                users.append(user)
                items.append(item)
                labels.append(label)

            yield torch.tensor(users, dtype=torch.long), torch.tensor(
                items, dtype=torch.long
            ), torch.tensor(labels, dtype=torch.float)


def load_sparse_matrix_from_csv(csv_file: str) -> coo_matrix:
    df = pd.read_csv(csv_file, index_col=0)
    row, col, data = [], [], []
    for user_id, row_data in df.iterrows():
        for game_id, owned in enumerate(row_data):
            if owned:
                row.append(user_id)
                col.append(game_id)
                data.append(owned)
    return coo_matrix((data, (row, col)), shape=df.shape)


def get_interaction_from_sparse_matrix(
    matrix: coo_matrix, idx: int
) -> Tuple[int, int, int]:
    return matrix.row[idx], matrix.col[idx], matrix.data[idx]


class NCF(nn.Module):
    def __init__(
        self, num_users: int, num_games: int, embedding_dim: int, dropout_rate: float
    ):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.game_embedding = nn.Embedding(num_games, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.fc3 = nn.Linear(embedding_dim // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(embedding_dim)
        self.batch_norm2 = nn.BatchNorm1d(embedding_dim // 2)

    def forward(self, user_ids: torch.Tensor, game_ids: torch.Tensor) -> torch.Tensor:
        user_embeds = self.user_embedding(user_ids)
        game_embeds = self.game_embedding(game_ids)
        concat_embeds = torch.cat((user_embeds, game_embeds), dim=1)
        dropout_embeds = self.dropout(concat_embeds)
        hidden1 = torch.relu(self.batch_norm1(self.fc1(dropout_embeds)))
        hidden2 = torch.relu(self.batch_norm2(self.fc2(hidden1)))
        hidden3 = torch.sigmoid(self.fc3(hidden2))
        return hidden3.squeeze()


def recommend_games(
    model: nn.Module, user_id: int, num_games: int, top_k: int
) -> np.ndarray:
    model.eval()
    user_ids = torch.tensor([user_id] * num_games).long()
    game_ids = torch.tensor(range(num_games)).long()
    with torch.no_grad():
        scores = model(user_ids, game_ids)
    top_game_indices = scores.argsort(descending=True)[:top_k].numpy()
    return top_game_indices


def train(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(data_loader, desc="Training", unit="batch")
    for user_indices, item_indices, labels in progress_bar:
        user_indices = user_indices.to(device).long()
        item_indices = item_indices.to(device).long()
        labels = labels.to(device).float()

        optimizer.zero_grad()
        predictions = model(user_indices, item_indices)
        loss = criterion(predictions.view(-1, 1), labels.view(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (len(progress_bar) + 1)})

    progress_bar.close()
    return running_loss / len(data_loader)


def precision_recall_at_k(
    predictions: Union[np.ndarray, List[float]],
    labels: Union[np.ndarray, List[int]],
    k: int,
) -> Tuple[float, float]:
    sorted_indices = np.argsort(predictions)
    top_k_indices = sorted_indices[-k:]

    labels_at_k = [labels[idx] for idx in top_k_indices]
    labels_at_k = np.array(labels_at_k)

    precision = labels_at_k.sum() / k
    recall = labels_at_k.sum() / np.sum(labels)
    return precision, recall


def evaluate(
    model: nn.Module,
    data_loader: data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    k: int,
) -> Tuple[float, float, float, float]:
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_precision_at_k = 0.0
    total_recall_at_k = 0.0
    total_samples = 0

    with torch.no_grad():
        for user_indices, item_indices, labels in data_loader:
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            labels = labels.to(device)

            predictions = model(user_indices, item_indices)
            loss = criterion(predictions.view(-1), labels.float())
            running_loss += loss.item()

            # Calculate accuracy
            total_correct += ((predictions.view(-1) > 0.5) == labels).sum().item()
            total_samples += labels.size(0)

            # Calculate Precision@K and Recall@K
            user_predictions = predictions.cpu().numpy()
            user_labels = labels.cpu().numpy()
            precision, recall = precision_recall_at_k(user_predictions, user_labels, k)
            total_precision_at_k += precision
            total_recall_at_k += recall

    average_loss = running_loss / len(data_loader)
    accuracy = total_correct / total_samples
    average_precision_at_k = total_precision_at_k / total_samples
    average_recall_at_k = total_recall_at_k / total_samples
    return average_loss, accuracy, average_precision_at_k, average_recall_at_k


class Autoencoder(nn.Module):
    def __init__(
        self, num_users: int, num_games: int, embedding_dim: int, dropout_rate: float
    ):
        super(Autoencoder, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, num_games),
            nn.Sigmoid()
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        user_embeds = self.user_embedding(user_ids)
        encoded = self.encoder(user_embeds)
        decoded = self.decoder(encoded)
        return decoded


def train(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(data_loader, desc="Training", unit="batch")
    for user_indices, labels in progress_bar:
        user_indices = user_indices.to(device).long()
        labels = labels.to(device).float()

        optimizer.zero_grad()
        predictions = model(user_indices)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (len(progress_bar) + 1)})

    progress_bar.close()
    return running_loss / len(data_loader)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for user_indices, labels in data_loader:
            user_indices = user_indices.to(device)
            labels = labels.to(device)

            predictions = model(user_indices)
            loss = criterion(predictions, labels)
            running_loss += loss.item()

            total_correct += ((predictions > 0.5) == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = running_loss / len(data_loader)
    accuracy = total_correct / total_samples
    return average_loss, accuracy




if __name__ == "__main__":
    # Save sparse matrix to file
    hdf5_file = "interactions.h5"
    df = pd.read_hdf(hdf5_file, key="df")
    #print("Matrix loading...")
    sparse_matrix_file = "sparse_matrix.npz"
    sparse_matrix = load_npz(sparse_matrix_file)
    #print("Matrix loaded!")

    # Split interactions into train and test sets
    #print("Data splitting...")
    num_interactions = sparse_matrix.nnz
    indices = np.arange(num_interactions)
    np.random.shuffle(indices)
    train_indices, test_indices = train_test_split(
        indices, test_size=0.15, random_state=42
    )

    train_interactions = [
        get_interaction_from_sparse_matrix(sparse_matrix, idx) for idx in train_indices
    ]
    test_interactions = [
        get_interaction_from_sparse_matrix(sparse_matrix, idx) for idx in test_indices
    ]
    #print("Data splitted!")

    # Create generator datasets and DataLoader
    batch_size = 64
    train_dataset = InteractionMatrixDataset(train_interactions)
    test_dataset = InteractionMatrixDataset(test_interactions)
    #print("Dataloader loading...")
    train_loader = InteractionMatrixDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = InteractionMatrixDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    #print(f"Dataloader loaded: {len(train_dataset)}, {len(test_dataset)} test")

    # Set up model parameters
    num_users = df.shape[0]
    num_games = df.shape[1]
    embedding_dim = 45
    dropout_rate = 0.35    
    model = NCF(num_users, num_games, embedding_dim, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Создание TensorBoard SummaryWriter
    writer = SummaryWriter(
        "logs/NCF " + datetime.datetime.now().strftime("%d-%m - %H-%M-%S")
    )
    # Train and evaluate model with additional metrics
    k = 2
    num_epochs = 10
    for epoch in range(num_epochs):
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, device)

        # Evaluate
        test_loss, accuracy, precision_at_k, recall_at_k = evaluate(
            model, test_loader, criterion, device, k
        )

        # Log loss and metrics in TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)
        writer.add_scalar("Precision@K", precision_at_k, epoch)
        writer.add_scalar("Recall@K", recall_at_k, epoch)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision@{k}: {precision_at_k:.4f}, Recall@{k}: {recall_at_k:.4f}"
        )

    # Close TensorBoard SummaryWriter
    writer.close()
