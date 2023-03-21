import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from resources import create_interaction_matrix
import warnings
from resources import *
import datetime
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics, losses, optimizers
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras import regularizers


warnings.filterwarnings("ignore")

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Log tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

interactions = create_interaction_matrix(
    df=pd.read_csv('recdata.csv', index_col=0).rename(columns={'variable': 'id', 'value': 'owned'}),
    user_col="uid", item_col="id", rating_col="owned"
)
num_users = interactions.shape[0]
num_items = interactions.shape[1]

# Split data into training and test sets
split_index = int(0.8 * num_users)
train_interactions = interactions[:split_index]
test_interactions = interactions[split_index:]


# Define the model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(num_items,)),
    Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
    Dense(num_items, activation='sigmoid')
])
loss = (losses.BinaryCrossentropy())
optimizer = optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss=loss)

model.fit(
    train_interactions,
    train_interactions,
    epochs=300,
    batch_size=16,
    validation_data=(test_interactions, test_interactions),
    callbacks=[tensorboard_callback]
)

# Predict the top k items for each user
k = 10
test_preds = model.predict(test_interactions)
top_k = np.argsort(test_preds, axis=1)[:, -k:]

# Compute precision and recall at k
test_labels = test_interactions[:, top_k]
precision = precision_score(test_labels.flatten(), np.ones_like(test_labels.flatten()))
recall = recall_score(test_labels.flatten(), np.ones_like(test_labels.flatten()))

print(f"Precision at k: {precision}")
print(f"Recall at k: {recall}")
