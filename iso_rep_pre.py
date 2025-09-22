import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import torch.nn.functional as F

EMBEDDING_DIM = 128
PROJECTION_DIM = 64
# 1. Load and Preprocess Data without Scikit-learn
def load_and_preprocess_data():
    """Loads and preprocesses the Adult Census Income dataset using pandas and numpy."""
    print("Fetching dataset...")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets

    # Drop the 'fnlwgt' column
    X = X.drop('fnlwgt', axis=1)

    # Convert target to binary (0 for <=50K, 1 for >50K)
    y = y['income'].apply(lambda x: 1 if x == '>50K' or x == '>50K.' else 0)

    # Replace '?' with NaN
    X.replace('?', np.nan, inplace=True)

    # Split data first to prevent data leakage
    print("Splitting data...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)

    # Make copies to avoid SettingWithCopyWarning
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    
    # Store the sensitive attributes from the raw training data *before* preprocessing
    X_train_sensitive = X_train[['race', 'sex']].copy()
    X_test_sensitive = X_test[['race', 'sex']].copy()

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # --- Preprocessing ---
    print("Applying preprocessing...")
    # Imputation
    # Calculate fillers from the training set ONLY
    num_fillers = X_train[numerical_features].median()
    cat_fillers = X_train[categorical_features].mode().iloc[0]

    # Apply to both train and test sets
    X_train[numerical_features] = X_train[numerical_features].fillna(num_fillers)
    X_test[numerical_features] = X_test[numerical_features].fillna(num_fillers)
    X_train[categorical_features] = X_train[categorical_features].fillna(cat_fillers)
    X_test[categorical_features] = X_test[categorical_features].fillna(cat_fillers)

    # Scaling numerical features
    # Calculate scaling factors from the training set ONLY
    mean = X_train[numerical_features].mean()
    std = X_train[numerical_features].std()
    
    X_train[numerical_features] = (X_train[numerical_features] - mean) / std
    X_test[numerical_features] = (X_test[numerical_features] - mean) / std

    # One-Hot Encoding for categorical features
    X_train = pd.get_dummies(X_train, columns=categorical_features, dummy_na=False)
    X_test = pd.get_dummies(X_test, columns=categorical_features, dummy_na=False)
    
    # Align columns - crucial for consistent feature sets
    # This ensures X_test has the exact same columns as X_train, filling missing ones with 0
    # and dropping any columns from X_test that were not in X_train.
    X_test = X_test.reindex(columns = X_train.columns, fill_value=0)

    # Return the sensitive attributes for the training set along with the processed data
    return (X_train.values, X_test.values, 
            y_train.values, y_test.values, 
            X_train_sensitive, X_test_sensitive)

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    """Manually splits pandas DataFrames into train and test sets."""
    if random_state:
        np.random.seed(random_state)
    
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    # Return the raw X data for the sensitive attributes split as well
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

# 2. Define the Models
class EmbeddingModel(nn.Module): #F
    """An MLP that creates a 128-dimensional embedding from the input features."""
    def __init__(self, input_dim):
        super(EmbeddingModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM) # Output embedding of size 128
        )

    def forward(self, x):
        embedding = self.layers(x)
        return F.normalize(embedding, p=2, dim=1)

class SensitiveAttributeClassifier(nn.Module): 
    """A simple MLP to predict a sensitive attribute from embeddings."""
    def __init__(self, embedding_dim, num_classes):
        super(SensitiveAttributeClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class PredictionHead(nn.Module): #H
    """An MLP that takes a 128-dimensional embedding and predicts income."""
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super(PredictionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class EmbeddingProjection(nn.Module): #G
    """An MLP that projects the 128D embedding to a 64D embedding."""
    def __init__(self, input_dim=EMBEDDING_DIM, output_dim=PROJECTION_DIM):
        super(EmbeddingProjection, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        projected_embedding = self.layers(x)
        return F.normalize(projected_embedding, p=2, dim=1)

class Codebook(nn.Module):
    """
    A class to store a dictionary of learnable embedding vectors for
    different demographic groups. All exposed embeddings are L2-normalized.
    """
    def __init__(self, group_keys, embedding_dim):
        """
        Initializes the Codebook.
        Args:
            group_keys (list): A list of strings, where each string is a unique
                               key for a demographic group (e.g., 'Male_White').
            embedding_dim (int): The dimensionality of the embedding vectors.
        """
        super(Codebook, self).__init__()
        self.group_keys = group_keys
        self.num_groups = len(group_keys)
        self.embedding_dim = embedding_dim
        
        # Map string keys to integer indices for nn.Embedding
        self.key_to_idx = {key: i for i, key in enumerate(self.group_keys)}
        
        # The codebook is an embedding layer where each index corresponds to a group
        self.embeddings = nn.Embedding(self.num_groups, self.embedding_dim)
        
        # Initialize the embeddings (optional, but good practice)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, indices):
        """Looks up and returns L2-normalized embeddings for the given indices."""
        raw_embeddings = self.embeddings(indices)
        return F.normalize(raw_embeddings, p=2, dim=1)

    def get_all_embeddings(self):
        """Returns all L2-normalized embedding vectors in the codebook."""
        return F.normalize(self.embeddings.weight, p=2, dim=1)

    def get_embedding_for_group(self, group_key):
        """Returns the L2-normalized embedding for a specific group key."""
        idx = self.key_to_idx[group_key]
        normalized_weights = F.normalize(self.embeddings.weight, p=2, dim=1)
        return normalized_weights[idx]

    def find_closest_embedding(self, input_embeddings):
        """
        Finds the closest codebook embedding for each embedding in the input batch.
        Args:
            input_embeddings (Tensor): A batch of L2-normalized embeddings.
        Returns:
            Tensor: The closest L2-normalized codebook vectors for each input embedding.
        """
        # Get L2-normalized codebook embeddings
        codebook_norm = self.get_all_embeddings()
        
        # Calculate cosine similarity (dot product of normalized vectors)
        cos_sim = torch.matmul(input_embeddings, codebook_norm.t())
        
        # Find the index of the most similar codebook vector for each input
        closest_indices = torch.argmax(cos_sim, dim=1)
        
        # Retrieve the normalized closest embeddings using the indices
        return self(closest_indices)

    def find_closest_embedding_indices(self, input_embeddings):
        """
        Finds the index of the closest codebook embedding for each embedding in the input batch.
        Args:
            input_embeddings (Tensor): A batch of L2-normalized embeddings.
        Returns:
            Tensor: The indices of the closest codebook vectors.
        """
        # Get L2-normalized codebook embeddings
        codebook_norm = self.get_all_embeddings()
        
        # Calculate cosine similarity (dot product of normalized vectors)
        cos_sim = torch.matmul(input_embeddings, codebook_norm.t())
        
        # Find the index of the most similar codebook vector for each input
        return torch.argmax(cos_sim, dim=1)


def train_phase1(embedding_model, prediction_head, embedding_projection, codebook, 
                 train_loader, criterion, optimizer, n_epochs, alpha, inter_group_margin,
                 train_embedding_model=True):
    """Trains the models in Phase 1 for metric learning."""
    print(f"\n--- Starting Phase 1: Training for Metric Learning ---")
    print(f"Starting training for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        if train_embedding_model:
            embedding_model.train()
        else:
            embedding_model.eval() # Keep dropout off if not training
        
        prediction_head.train()
        embedding_projection.train()
        codebook.train()

        running_loss = 0.0
        for inputs, labels, groups in train_loader:
            optimizer.zero_grad()
            
            # --- Main Forward Pass ---
            embeddings = embedding_model(inputs)
            projected_embeddings = embedding_projection(embeddings)
            outputs = prediction_head(embeddings)
            
            # 1. Main prediction loss (on all samples), H
            prediction_loss = criterion(outputs, labels) if train_head else 0.0

            # --- Metric Learning Loss (on filtered samples) ---, G
            mask = groups != -1
            iso_loss = 0.0
            if mask.sum() > 0:
                filtered_projected_embs = projected_embeddings[mask]
                filtered_groups = groups[mask]
                target_centroids = codebook(filtered_groups)
                pull_loss = F.mse_loss(filtered_projected_embs, target_centroids)
                codebook_centroids = codebook.get_all_embeddings()
                pairwise_dist = torch.pdist(codebook_centroids, p=2)
                push_loss = torch.relu(inter_group_margin - pairwise_dist).mean()
                iso_loss = pull_loss + push_loss
            
            # --- Combine Losses and Backpropagate ---
            total_loss = prediction_loss + (alpha * iso_loss)
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")

def train_phase2(embedding_model, prediction_head, embedding_projection, codebook,
                 train_loader, criterion, optimizer, n_epochs, beta):
    """Trains the models in Phase 2 for disentanglement."""
    print(f"\n--- Starting Phase 2: Training for Disentanglement ---")
    
    # Freeze the embedding_projection and codebook
    for param in embedding_projection.parameters():
        param.requires_grad = False
    for param in codebook.parameters():
        param.requires_grad = False
    
    print(f"Starting training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        embedding_model.train()
        prediction_head.train()
        
        running_loss_phase2 = 0.0
        for inputs, labels, groups in train_loader:
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            embeddings = embedding_model(inputs)
            projected_embeddings = embedding_projection(embeddings)
            outputs = prediction_head(embeddings)
            
            # --- Loss Calculation ---
            prediction_loss = criterion(outputs, labels) if train_head else 0.0
            codebook_centroids = codebook.get_all_embeddings()
            distances = torch.cdist(projected_embeddings, codebook_centroids, p=2)

            max_distances, _ = torch.max(distances, dim=1)
            min_distances, _ = torch.min(distances, dim=1)
            disentanglement_loss = (max_distances - min_distances).mean()
            
            # Calculate the average distance for each projected_embedding across all centroids
            #average_distances = distances.mean(dim=1, keepdim=True)
            
            # Calculate the absolute difference between each distance and its corresponding average
            #differences = torch.abs(distances - average_distances)
            
            # Sum all the differences to get the disentanglement_loss
            #disentanglement_loss = differences.sum()
            
            # --- Combine Losses and Backpropagate ---
            total_loss_phase2 = prediction_loss + (beta * disentanglement_loss)
            
            total_loss_phase2.backward()
            optimizer.step()
            running_loss_phase2 += total_loss_phase2.item()
            
        epoch_loss_phase2 = running_loss_phase2 / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss_phase2:.4f}")

def evaluate_information_leakage(embeddings, sensitive_attributes):
    """
    Trains a classifier to predict sensitive attributes from embeddings.
    A lower accuracy indicates better fairness.
    """
    print("\n--- Evaluating Intersectional Information Leakage ---")
    
    # We will evaluate for the intersection of 'sex' and 'race'.
    # Create the intersectional group key
    sensitive_attributes['group_key'] = sensitive_attributes['sex'].astype(str) + '_' + sensitive_attributes['race'].astype(str)
    
    # Filter for the 4 groups of interest
    groups_of_interest = ['Male_White', 'Male_Black', 'Female_White', 'Female_Black']
    mask = sensitive_attributes['group_key'].isin(groups_of_interest)
    
    filtered_embeddings = embeddings[mask.values]
    filtered_attributes = sensitive_attributes[mask]

    if len(filtered_attributes) == 0:
        print("  - No samples from the specified intersectional groups found to evaluate information leakage.")
        return

    print(f"Training classifier to predict intersectional group...")
    
    # Prepare data
    labels = filtered_attributes['group_key'].astype('category').cat.codes.values
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    num_classes = len(filtered_attributes['group_key'].unique())

    dataset = TensorDataset(filtered_embeddings.detach(), labels_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train classifier
    classifier = SensitiveAttributeClassifier(filtered_embeddings.shape[1], num_classes)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Simple training loop for the attribute classifier
    for epoch in range(5): # 5 epochs should be enough
        for emb_batch, label_batch in loader:
            optimizer.zero_grad()
            outputs = classifier(emb_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()

    # Evaluate classifier
    with torch.no_grad():
        outputs = classifier(filtered_embeddings.detach())
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels_tensor).float().mean().item()
        print(f"  - Accuracy of predicting intersectional group from embeddings: {accuracy:.4f}")

def calculate_group_fairness_metrics(y_true, y_pred, sensitive_attributes):
    """Calculates Demographic Parity and Equalized Odds for intersectional groups."""
    print("\n--- Calculating Group Fairness Metrics ---")
    
    # This was already done in the information leakage function, but we do it again in case this function is called independently.
    sensitive_attributes['group_key'] = sensitive_attributes['sex'].astype(str) + '_' + sensitive_attributes['race'].astype(str)

    df = pd.DataFrame({
        'y_true': y_true.squeeze().numpy(),
        'y_pred': y_pred.squeeze().numpy(),
        'group_key': sensitive_attributes['group_key']
    })

    # Filter for the 4 groups of interest
    groups_of_interest = ['Male_White', 'Male_Black', 'Female_White', 'Female_Black']
    df = df[df['group_key'].isin(groups_of_interest)]

    if df.empty:
        print("  - No samples from the specified intersectional groups found to calculate fairness metrics.")
        return

    attribute = 'group_key'
    print(f"Metrics for attribute: '{attribute}'")
    groups = df[attribute].unique()
    
    # Demographic Parity
    positive_rates = {group: df[df[attribute] == group]['y_pred'].mean() for group in groups}
    print("  - Demographic Parity (Positive Prediction Rate):")
    for group, rate in sorted(positive_rates.items()):
        print(f"    - {group}: {rate:.4f}")
    
    if len(positive_rates) > 1:
        rates = list(positive_rates.values())
        dp_mean = np.mean(rates)
        dp_std = np.std(rates)
        print(f"    - Mean: {dp_mean:.6f}, Std Dev: {dp_std:.6f}")

    # Equalized Odds
    print("  - Equalized Odds (True Positive Rate & False Positive Rate):")
    tprs = {}
    fprs = {}
    for group in sorted(groups):
        group_df = df[df[attribute] == group]
        # Calculate TPR, handling cases where there are no positive samples
        if (group_df['y_true'] == 1).sum() > 0:
            tpr = (group_df['y_pred'][group_df['y_true'] == 1] == 1).mean()
        else:
            tpr = float('nan')
        # Calculate FPR, handling cases where there are no negative samples
        if (group_df['y_true'] == 0).sum() > 0:
            fpr = (group_df['y_pred'][group_df['y_true'] == 0] == 1).mean()
        else:
            fpr = float('nan')
        
        tprs[group] = tpr
        fprs[group] = fpr
        print(f"    - {group}: TPR={tpr:.4f}, FPR={fpr:.4f}")

    if len(tprs) > 1:
        tpr_values = list(tprs.values())
        tpr_mean = np.nanmean(tpr_values)
        tpr_std = np.nanstd(tpr_values)
        print(f"    - TPR Mean: {tpr_mean:.6f}, Std Dev: {tpr_std:.6f}")
    if len(fprs) > 1:
        fpr_values = list(fprs.values())
        fpr_mean = np.nanmean(fpr_values)
        fpr_std = np.nanstd(fpr_values)
        print(f"    - FPR Mean: {fpr_mean:.6f}, Std Dev: {fpr_std:.6f}")


def evaluate_mutual_information(embeddings, sensitive_attributes):
    """
    Evaluates the mutual information between embeddings and sensitive attributes.
    A lower MI indicates better fairness.
    """
    print("\n--- Evaluating Mutual Information ---")
    
    # Create the intersectional group key
    sensitive_attributes['group_key'] = sensitive_attributes['sex'].astype(str) + '_' + sensitive_attributes['race'].astype(str)
    
    # Filter for the 4 groups of interest
    groups_of_interest = ['Male_White', 'Male_Black', 'Female_White', 'Female_Black']
    mask = sensitive_attributes['group_key'].isin(groups_of_interest)
    
    filtered_embeddings = embeddings[mask.values]
    filtered_attributes = sensitive_attributes[mask]

    if len(filtered_attributes) < 2: # Need at least 2 samples for MI calculation
        print("  - Not enough samples from the specified intersectional groups found to evaluate mutual information.")
        return

    # Prepare data
    labels = filtered_attributes['group_key'].astype('category').cat.codes.values
    
    # --- Custom Mutual Information Calculation using Numpy ---
    embeddings_np = filtered_embeddings.detach().cpu().numpy()
    
    total_mi = 0.0
    n_samples = embeddings_np.shape[0]
    n_bins = int(np.sqrt(n_samples / 5)) # Sturges' rule variation for binning
    n_bins = max(n_bins, 2) # Ensure at least 2 bins
    
    for i in range(embeddings_np.shape[1]): # Iterate over each embedding dimension
        # Discretize the continuous embedding feature
        embedding_feature = embeddings_np[:, i]
        bins = np.linspace(embedding_feature.min(), embedding_feature.max(), n_bins + 1)
        binned_feature = np.digitize(embedding_feature, bins) - 1

        # Calculate joint and marginal probabilities
        joint_prob = np.histogram2d(binned_feature, labels, bins=[n_bins, len(np.unique(labels))])[0] / n_samples
        marginal_feature = np.sum(joint_prob, axis=1)
        marginal_labels = np.sum(joint_prob, axis=0)

        # Calculate mutual information for this feature
        mi = 0.0
        for row in range(n_bins):
            for col in range(len(np.unique(labels))):
                if joint_prob[row, col] > 0 and marginal_feature[row] > 0 and marginal_labels[col] > 0:
                    mi += joint_prob[row, col] * np.log(joint_prob[row, col] / (marginal_feature[row] * marginal_labels[col]))
        
        total_mi += mi
    
    print(f"  - Total Mutual Information between embeddings and intersectional group: {total_mi:.6f}")


def evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                   X_test_tensor, y_test_tensor, X_test_sensitive,
                   filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive):
    """Evaluates the model for income prediction and group prediction accuracy."""
    print("\n--- Evaluating Model ---")
    embedding_model.eval()
    prediction_head.eval()
    if embedding_projection:
        embedding_projection.eval()
    if codebook:
        codebook.eval()

    # 1. Income Prediction Accuracy
    with torch.no_grad():
        embeddings = embedding_model(X_test_tensor)
        y_pred = prediction_head(embeddings)
        predicted = (y_pred > 0.5).float()
        accuracy = (predicted == y_test_tensor).float().mean()
        print(f"Accuracy on the test set: {accuracy.item():.4f}")

    # --- New Fairness Evaluations ---
    # Convert sensitive attributes to categorical for easier processing
    X_test_sensitive_eval = X_test_sensitive.copy()
    
    evaluate_information_leakage(embeddings, X_test_sensitive_eval.copy())
    calculate_group_fairness_metrics(y_test_tensor, predicted, X_test_sensitive_eval.copy())
    evaluate_mutual_information(embeddings, X_test_sensitive_eval.copy())
    # --- End of New Evaluations ---


    # 2. Metric Learning Accuracy
    print("\nEvaluating metric learning for group prediction...")
    metric_accuracy = None
    if codebook and embedding_projection and len(filtered_X_test_sensitive) > 0:
        with torch.no_grad():
            base_embeddings = embedding_model(filtered_X_test_tensor)
            projected_embeddings = embedding_projection(base_embeddings)
            predicted_group_indices = codebook.find_closest_embedding_indices(projected_embeddings)
            metric_accuracy_tensor = (predicted_group_indices == true_group_indices).float().mean()
            metric_accuracy = metric_accuracy_tensor.item()
            print(f"Accuracy in predicting Race/Sex group on the test set (Black/White only): {metric_accuracy:.4f}")
    else:
        print("Skipping metric learning evaluation (no codebook/projection or no filtered data).")
    
    return metric_accuracy

def analyze_codebook(codebook, group_keys):
    """Prints analysis of the learned codebook centroids."""
    print("\n--- Learned Codebook Analysis ---")
    with torch.no_grad():
        codebook_centroids = codebook.get_all_embeddings()
        
        print("L2 Norm (Length) of each group centroid:")
        for i, key in enumerate(group_keys):
            l2_length = torch.norm(codebook_centroids[i], p=2)
            print(f"  - {key}: {l2_length.item():.4f}")
            
        print("\nPairwise L2 Distances between group centroids:")
        pairwise_dist = torch.pdist(codebook_centroids, p=2)
        
        distance_idx = 0
        for i in range(len(group_keys)):
            for j in range(i + 1, len(group_keys)):
                dist = pairwise_dist[distance_idx]
                print(f"  - Distance({group_keys[i]}, {group_keys[j]}): {dist.item():.4f}")
                distance_idx += 1

# 3. Train and Evaluate the Model
if __name__ == "__main__":
    seed = 9973
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_head = True #False #True

    # Load data
    X_train, X_test, y_train, y_test, X_train_sensitive, X_test_sensitive = load_and_preprocess_data()

    # --- Prepare sensitive attribute data for the training loop ---
    # Create the group key strings (e.g., 'Male_White')
    X_train_sensitive['group_key'] = X_train_sensitive['sex'] + '_' + X_train_sensitive['race']
    
    # Convert group keys to integer labels for the dataset
    group_keys = ['Male_White', 'Male_Black', 'Female_White', 'Female_Black']
    key_to_idx = {key: i for i, key in enumerate(group_keys)}
    
    # Map keys to indices, with -1 for groups we want to ignore
    group_indices = X_train_sensitive['group_key'].map(key_to_idx).fillna(-1).astype(int)

    # Convert data to PyTorch Tensors
    #X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    try:
        X_train_clean = X_train.astype(np.float32)
        X_train_tensor = torch.tensor(X_train_clean)
    except ValueError as e:
        print(f"Error during type conversion: {e}")
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    group_indices_tensor = torch.tensor(group_indices.values, dtype=torch.long)
    #X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    try:
        X_test_clean = X_test.astype(np.float32)
        X_test_tensor = torch.tensor(X_test_clean)
    except ValueError as e:
        print(f"Error during type conversion: {e}")
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, group_indices_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    # Initialize model, loss, and optimizer
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    # --- Baseline Model Training & Evaluation ---
    print("\n\n--- Training and Evaluating Baseline Model (No Fairness Intervention) ---")
    baseline_embedding_model = EmbeddingModel(input_dim)
    baseline_prediction_head = PredictionHead()
    
    baseline_params = list(baseline_embedding_model.parameters()) + list(baseline_prediction_head.parameters() if train_head else [])
    baseline_optimizer = optim.Adam(baseline_params, lr=0.001)
    baseline_criterion = nn.BCELoss()

    # Simple training loop for the baseline model
    for epoch in range(20): # Using 20 epochs similar to other phases
        baseline_embedding_model.train()
        baseline_prediction_head.train() if train_head else baseline_prediction_head.eval()
        running_loss = 0.0
        # Create a simple dataloader without group info for this training
        baseline_train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        baseline_train_loader = DataLoader(dataset=baseline_train_dataset, batch_size=128, shuffle=True)
        
        for inputs, labels in baseline_train_loader:
            baseline_optimizer.zero_grad()
            embeddings = baseline_embedding_model(inputs)
            outputs = baseline_prediction_head(embeddings)
            loss = baseline_criterion(outputs, labels)
            loss.backward()
            baseline_optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/20, Baseline Model Loss: {running_loss/len(baseline_train_loader):.4f}")

    # For baseline evaluation, we don't have a trained codebook or projection, so we pass None
    # We create dummy tensors for the filtered data as they are not used for the main evaluation part
    dummy_filtered_tensor = torch.empty(0, X_test.shape[1])
    dummy_indices_tensor = torch.empty(0, dtype=torch.long)
    dummy_sensitive_df = pd.DataFrame(columns=X_test_sensitive.columns)

    evaluate_model(baseline_embedding_model, baseline_prediction_head, None, None,
                   torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1), X_test_sensitive,
                   dummy_filtered_tensor, dummy_indices_tensor, dummy_sensitive_df)
    
    # --- End of Baseline Evaluation ---


    embedding_model = EmbeddingModel(input_dim)
    prediction_head = PredictionHead()
    embedding_projection = EmbeddingProjection()
    codebook = Codebook(group_keys=group_keys, embedding_dim=PROJECTION_DIM)

    # --- Phase 1 Training ---
    combined_params = (
        list(embedding_model.parameters()) + 
        list(prediction_head.parameters() if train_head else []) +
        list(embedding_projection.parameters()) +
        list(codebook.parameters())
    )
    criterion = nn.BCELoss()
    optimizer_phase1 = optim.Adam(combined_params, lr=0.001)
    
    train_phase1(embedding_model, prediction_head, embedding_projection, codebook,
                 train_loader, criterion, optimizer_phase1, n_epochs=20, alpha=0.5, inter_group_margin=1.0)

    # --- Evaluation after Phase 1 ---
    # Prepare filtered test data for evaluation
    X_test_sensitive['group_key'] = X_test_sensitive['sex'] + '_' + X_test_sensitive['race']
    test_mask = X_test_sensitive['group_key'].isin(group_keys)
    filtered_X_test_sensitive = X_test_sensitive[test_mask]
    filtered_X_test_tensor = X_test_tensor[test_mask.values]
    true_group_indices = torch.tensor(
        filtered_X_test_sensitive['group_key'].map(key_to_idx).values, 
        dtype=torch.long
    )

    evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                   X_test_tensor, y_test_tensor, X_test_sensitive,
                   filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive)
    
    #analyze_codebook(codebook, group_keys)

    # --- Phase 2 Training ---
    optimizer_phase2 = optim.Adam(
        list(embedding_model.parameters()) + list(prediction_head.parameters() if train_head else []), 
        lr=0.001
    )
    
    train_phase2(embedding_model, prediction_head, embedding_projection, codebook,
                 train_loader, criterion, optimizer_phase2, n_epochs=20, beta=0.3)

    # --- Evaluation after Phase 2 ---
    evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                   X_test_tensor, y_test_tensor, X_test_sensitive,
                   filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive)

    #analyze_codebook(codebook, group_keys)

    # --- Iterative Adversarial Training ---
    n_iterations = 5 #30 #15
    iterative_epochs = 5 # Fewer epochs for each iterative step (originally 5) 10
    patience = 30 #3
    patience_counter = 0
    best_metric_accuracy = float('inf')

    for i in range(n_iterations):
        print(f"\n\n--- Adversarial Iteration {i+1}/{n_iterations} ---")

        # --- Step 1: Train the adversary (projection and codebook) ---
        print("\nTraining the adversary (Phase 1 variant)...")
        # Freeze embedding_model, unfreeze the rest
        for param in embedding_model.parameters(): param.requires_grad = False
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = True
        for param in codebook.parameters(): param.requires_grad = True

        optimizer_iter_1 = optim.Adam(
            list(prediction_head.parameters() if train_head else []) +
            list(embedding_projection.parameters()) +
            list(codebook.parameters()),
            lr=0.001
        )
        
        train_phase1(embedding_model, prediction_head, embedding_projection, codebook,
                     train_loader, criterion, optimizer_iter_1, n_epochs=iterative_epochs, 
                     alpha=0.5, inter_group_margin=1.0, train_embedding_model=False)

        print(f"\n--- Evaluation after training adversary in Iteration {i+1} ---")
        metric_accuracy = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                                         X_test_tensor, y_test_tensor, X_test_sensitive,
                                         filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive)

        # Early stopping logic
        if metric_accuracy is not None:
            if metric_accuracy < best_metric_accuracy:
                best_metric_accuracy = metric_accuracy
                patience_counter = 0
                print(f"  (New best metric accuracy: {best_metric_accuracy:.4f}. Patience counter reset.)")
            else:
                patience_counter += 1
                print(f"  (Metric accuracy did not decrease. Patience counter: {patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered: Metric accuracy has not decreased for {patience} iterations.")
                break

        # --- Step 2: Train the main model to fool the adversary ---
        print("\nTraining the main model to fool the adversary (Phase 2)...")
        # Unfreeze embedding_model and prediction_head for training
        for param in embedding_model.parameters(): param.requires_grad = True
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        
        optimizer_iter_2 = optim.Adam(
            list(embedding_model.parameters()) +
            list(prediction_head.parameters() if train_head else []),
            lr=0.001
        )
        
        train_phase2(embedding_model, prediction_head, embedding_projection, codebook,
                     train_loader, criterion, optimizer_iter_2, n_epochs=iterative_epochs, beta=0.3)
        
        # --- Evaluation after Phase 2 ---
        print(f"\n--- Evaluation after Phase 2 training main model in Iteration {i+1} ---")
        evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                     X_test_tensor, y_test_tensor, X_test_sensitive,
                     filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive)

    # --- Final Evaluation after Iterative Training ---
    print("\n\n--- Final Evaluation after Iterative Training ---")
    evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                   X_test_tensor, y_test_tensor, X_test_sensitive,
                   filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive)

    analyze_codebook(codebook, group_keys)
