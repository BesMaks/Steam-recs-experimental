import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from resources import *
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

def set_TB():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d-%m - %H-%M-%S")
    writer = SummaryWriter(log_dir=log_dir)
    return writer
def set_cuda():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n\nUsing device: {device}\n\n")
    return device

# interactions = create_interaction_matrix(
#     df=pd.read_csv('recdata.csv', index_col=0).rename(columns={'variable': 'id', 'value': 'owned'}),
#     user_col="uid", item_col="id", rating_col="owned"
# )


# Define the PyTorch model
class Recommender(nn.Module):
    def __init__(self, num_items, reg_lambda=0.01):
        super(Recommender, self).__init__()
        self.layer1 = nn.Linear(num_items, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_items)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.reg_lambda = reg_lambda

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

    def loss_fn(self, output, target):
        mse_loss = nn.functional.mse_loss(output, target, reduction='mean')
        l2_reg = torch.tensor(0.).cuda()
        for param in self.parameters():
            l2_reg += torch.norm(param)
        loss = mse_loss + self.reg_lambda * l2_reg
        return loss


class DMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64, reg_lambda=0.01):
        super(DMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.reg_lambda = reg_lambda

    def forward(self, user_idx, item_idx):
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        x = torch.cat([user_embed, item_embed], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze()

    def l2_regularization(self):
        l2_reg = torch.tensor(0.).to(self.user_embedding.weight.device)
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return l2_reg * self.reg_lambda


import torch
from torch.utils.data import Dataset

class InteractionDataset(Dataset):
    def __init__(self):
        # Load interaction matrix from csv file
        self.interaction_matrix = create_interaction_matrix(
            df=pd.read_csv('recdata.csv', index_col=0).rename(columns={'variable': 'id', 'value': 'owned'}),
            user_col="uid", item_col="id", rating_col="owned"
        )
        self.num_users = self.interaction_matrix.shape[0]
        self.num_items = self.interaction_matrix.shape[1]
    
    def __len__(self):
        # Return number of samples in the dataset
        return self.interaction_matrix.shape[0] * self.interaction_matrix.shape[1]
    
    def __getitem__(self, index):
        # Compute the row and column indices from the linear index
        row_idx = index // self.interaction_matrix.shape[1]
        col_idx = index % self.interaction_matrix.shape[1]
        
        # Retrieve the user and item IDs from the row and column indices
        user_id = torch.tensor(row_idx)
        item_id = torch.tensor(col_idx)
        
        # Retrieve the interaction value from the interaction matrix
        interaction = self.interaction_matrix[row_idx, col_idx]
        
        # Return the user ID, item ID, and interaction value as PyTorch tensors
        return user_id, item_id, interaction

def get_df_sizes():
    matrix = create_interaction_matrix(
            df=pd.read_csv('recdata.csv', index_col=0).rename(columns={'variable': 'id', 'value': 'owned'}),
            user_col="uid", item_col="id", rating_col="owned"
            )
    return matrix.shape[0], matrix.shape[1]   



# Get dataset
interactions = DataLoader(InteractionDataset(), batch_size=64, shuffle=True)
num_users, num_items = get_df_sizes()

# Define train and test dataset sizes
train_size = int(0.8 * len(interactions))
test_size = len(interactions) - train_size
train_dataset, test_dataset = random_split(interactions, [train_size, test_size])

# Create data loaders for train and test sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


writer = set_TB()
device = set_cuda()

model = DMF(num_users, num_items, embedding_dim=32, hidden_dim=64, reg_lambda=0.01).cuda().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# training loop
epochs = 100
for epoch in range(epochs):
    for i, (user_idx, item_idx, rating) in enumerate(train_loader.dataset):
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        rating = rating.to(device)
        
        optimizer.zero_grad()
        predictions = model(user_idx, item_idx)
        loss = nn.MSELoss()(predictions, rating) + model.l2_regularization()
        loss.backward()
        optimizer.step()

        # log training loss and model parameters to tensorboard
        if i % 100 == 0:
            writer.add_scalar('training_loss', loss.item(), epoch*len(train_loader)+i)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch*len(train_loader)+i)

# close tensorboard writer
writer.close()
def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        all_precision = []
        all_recall = []
        for user_idx, item_idx, rating in test_loader.dataset:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating = rating.to(device)
            
            # Get top k predictions
            predictions = model(user_idx, item_idx)
            _, topk_items = torch.topk(predictions, k=5)
            
            # Convert to numpy arrays
            topk_items = topk_items.cpu().numpy()
            actual_items = item_idx.cpu().numpy()
            
            # Compute precision and recall
            precision = precision_score(actual_items, topk_items, average='micro')
            recall = recall_score(actual_items, topk_items, average='micro')
            
            all_precision.append(precision)
            all_recall.append(recall)
            
        mean_precision = sum(all_precision) / len(all_precision)
        mean_recall = sum(all_recall) / len(all_recall)
        
        return mean_precision, mean_recall

precision, recall = evaluate(model, test_loader)
print(f"Test set precision@5: {precision}, recall@5: {recall}")





# # Train the model
# epochs = 150
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for i in range(0, train_interactions.shape[0], 16):
#         inputs = torch.from_numpy(train_interactions[i:i+16]).float().to(device)  # Move inputs to GPU and convert to float
#         targets = inputs.clone().to(device)  # Move targets to GPU
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * inputs.size(0)
#     train_loss = running_loss / train_interactions.shape[0]

#     model.eval()
#     with torch.no_grad():
#         test_inputs = torch.from_numpy(test_interactions).float().to(device)  # Move inputs to GPU and convert to float
#         test_targets = test_inputs.clone().to(device)  # Move targets to GPU
#         test_outputs = model(test_inputs)
#         test_loss = criterion(test_outputs, test_targets).item()

#         test_outputs_np = test_outputs.detach().cpu().numpy()  # Move outputs to CPU
#         test_targets_np = test_targets.cpu().numpy()  # Move targets to CPU
#         precision = precision_score(np.round(test_outputs_np), test_targets_np, average='micro')
#         recall = recall_score(np.round(test_outputs_np), test_targets_np, average='micro')

#     # Log metrics to TensorBoard
#     writer.add_scalar('Loss/Train', train_loss, epoch)
#     writer.add_scalar('Loss/Test', test_loss, epoch)
#     writer.add_scalar('Precision@k', precision, epoch)
#     writer.add_scalar('Recall@k', recall, epoch)

#     print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Precision@k: {precision:.4f}, Recall@k: {recall:.4f}")

# # Close TensorBoard writer
# writer.close()
