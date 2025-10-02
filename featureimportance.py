import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd

# 1. Load and Preprocess data
def load_and_preprocess_data():
    """Loads Adult dataset and preprocesses without one-hot encoding."""
    print("Fetching dataset...")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets

    # Drop 'fnlwgt' 
    X = X.drop('fnlwgt', axis=1)

    # Convert target to binary
    y = y['income'].apply(lambda x: 1 if x == '>50K' or x == '>50K.' else 0)

    # Replace '?' with NaN
    X.replace('?', np.nan, inplace=True)

    # Manual train/test split
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)

    # Sensitive attributes
    X_train_sensitive = X_train[['race', 'sex']].copy()
    X_test_sensitive = X_test[['race', 'sex']].copy()

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Impute missing values
    num_fillers = X_train[numerical_features].median()
    cat_fillers = X_train[categorical_features].mode().iloc[0]

    X_train[numerical_features] = X_train[numerical_features].fillna(num_fillers)
    X_test[numerical_features] = X_test[numerical_features].fillna(num_fillers)
    X_train[categorical_features] = X_train[categorical_features].fillna(cat_fillers)
    X_test[categorical_features] = X_test[categorical_features].fillna(cat_fillers)

    # Scale numerical features
    mean = X_train[numerical_features].mean()
    std = X_train[numerical_features].std()
    X_train[numerical_features] = (X_train[numerical_features] - mean) / std
    X_test[numerical_features] = (X_test[numerical_features] - mean) / std

    return X_train, X_test, y_train, y_test, X_train_sensitive, X_test_sensitive, numerical_features, categorical_features

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    """Manually splits pandas DataFrames into train and test sets."""
    if random_state:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

# 2. Define model
class PredictionHead(nn.Module):
    """MLP with categorical embeddings."""
    def __init__(self, numerical_dim, categorical_cardinalities, embedding_dim=4, hidden_dim=64):
        super(PredictionHead, self).__init__()
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim) for num_categories in categorical_cardinalities
        ])
        self.embedding_dim = embedding_dim
        self.numerical_dim = numerical_dim

        # MLP
        total_input_dim = numerical_dim + embedding_dim * len(categorical_cardinalities)
        self.layers = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_num, x_cat):
        # x_cat: (batch_size, num_categorical)
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat_emb = torch.cat(embedded, dim=1)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.layers(x)

# 3. Compute feature importance
def compute_feature_importance(X_train, y_train, X_test, numerical_features, categorical_features):
    """Computes per-feature importance using MAD of predictions."""
    # Map categories to integers
    cat_mappings = {}
    for col in categorical_features:
        categories = X_train[col].unique().tolist()
        cat_mappings[col] = {cat: idx for idx, cat in enumerate(categories)}
        X_train[col] = X_train[col].map(cat_mappings[col])
        X_test[col] = X_test[col].map(cat_mappings[col])

    # Convert to tensors
    X_train_num = torch.tensor(X_train[numerical_features].values.astype(np.float32))
    X_test_num = torch.tensor(X_test[numerical_features].values.astype(np.float32))
    X_train_cat = torch.tensor(X_train[categorical_features].values.astype(np.int64))
    X_test_cat = torch.tensor(X_test[categorical_features].values.astype(np.int64))
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).unsqueeze(1)

    # Define model
    categorical_cardinalities = [len(cat_mappings[col]) for col in categorical_features]
    model = PredictionHead(numerical_dim=len(numerical_features),
                           categorical_cardinalities=categorical_cardinalities)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    n_epochs = 20
    batch_size = 128
    train_dataset = TensorDataset(X_train_num, X_train_cat, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("\n--- Training Prediction Model ---")
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for x_num_batch, x_cat_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_num_batch, x_cat_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Feature importance
    print("\n--- Computing Feature Importance ---")
    feature_importance = {}

    # Numerical features
    for i, feat in enumerate(numerical_features):
        predictions = []
        for val in np.unique(X_train[feat]):
            x_num_mod = X_test_num.clone()
            x_num_mod[:, i] = val
            with torch.no_grad():
                pred = model(x_num_mod, X_test_cat).numpy()
                predictions.append(pred)
        predictions = np.concatenate(predictions, axis=1)
        mad = np.mean(np.abs(predictions - predictions.mean(axis=1, keepdims=True)))
        feature_importance[feat] = mad

    # Categorical features
    for i, feat in enumerate(categorical_features):
        predictions = []
        for val in range(len(cat_mappings[feat])):
            x_cat_mod = X_test_cat.clone()
            x_cat_mod[:, i] = val
            with torch.no_grad():
                pred = model(X_test_num, x_cat_mod).numpy()
                predictions.append(pred)
        predictions = np.concatenate(predictions, axis=1)
        mad = np.mean(np.abs(predictions - predictions.mean(axis=1, keepdims=True)))
        feature_importance[feat] = mad

    # Normalize importance
    total = sum(feature_importance.values())
    for k in feature_importance:
        feature_importance[k] /= total
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return feature_importance

# 4. Main
if __name__ == "__main__":
    seed = 9973
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test, X_train_sensitive, X_test_sensitive, numerical_features, categorical_features = load_and_preprocess_data()
    feature_importance = compute_feature_importance(X_train, y_train, X_test, numerical_features, categorical_features)

    print("\n--- Feature Importance (normalized) ---")
    for feat, imp in feature_importance.items():
        print(f"{feat}: {imp:.4f}")