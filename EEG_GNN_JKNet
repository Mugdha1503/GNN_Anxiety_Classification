import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import h5py
import scipy.signal as signal
from scipy.stats import entropy
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool
from torch_geometric.loader import DataLoader

# Define frequency bands
theta_band = (4, 7)    # Theta band: 4–7 Hz
beta_band = (13, 30)   # Beta band: 13–30 Hz
alpha_band = (8, 12)   # Alpha band: 8–12 Hz

# List of channels to extract
channels_of_interest = ['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'P7', 'P8']

# Map channel indices for the 8 channels
channel_indices = {'AF3': 0, 'AF4': 1, 'F3': 2, 'F4': 3, 'FC5': 4, 'FC6': 5, 'P7': 10, 'P8': 11}

def bandpass_filter(eeg_signal, fs, band):
    """Apply bandpass filter to EEG signal"""
    nyquist = 0.5 * fs
    low, high = band
    low /= nyquist
    high /= nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.lfilter(b, a, eeg_signal)

def compute_psd(eeg_signal, fs):
    """Compute Power Spectral Density"""
    freqs, psd = signal.welch(eeg_signal, fs, nperseg=256)
    return freqs, psd

def compute_cpcc_psd_beta(segment, fs=128):
    """
    Compute Correlation-based Pearson Correlation Coefficient (CPCC)
    using filtered PSD_Beta signals for each channel.
    """
    num_channels = len(channels_of_interest)
    cpcc_matrix = np.zeros((num_channels, num_channels))

    # Compute and filter PSD_Beta signals for each channel
    psd_beta_signals = []
    for ch_name, ch_index in channel_indices.items():
        if ch_name in channels_of_interest:
            # Extract channel signal
            eeg_signal = segment[ch_index, :]

            # Apply beta band filter
            beta_signal = bandpass_filter(eeg_signal, fs, beta_band)

            # Compute PSD for beta signal
            freqs, psd_beta = compute_psd(beta_signal, fs)
            psd_beta_signals.append(psd_beta)

    # Compute correlation between PSD_Beta signals
    for i in range(num_channels):
        for j in range(num_channels):
            cpcc_matrix[i, j] = np.corrcoef(psd_beta_signals[i], psd_beta_signals[j])[0, 1]

    return cpcc_matrix

def extract_features(segment, fs=128):
    """Extract comprehensive features from EEG segment"""
    features = {}

    for ch_name, ch_index in channel_indices.items():
        if ch_name not in channels_of_interest:
            continue

        eeg_signal = segment[ch_index, :]

        # Apply bandpass filters
        theta_signal = bandpass_filter(eeg_signal, fs, theta_band)
        beta_signal = bandpass_filter(eeg_signal, fs, beta_band)
        alpha_signal = bandpass_filter(eeg_signal, fs, alpha_band)

        # Compute PSDs
        _, psd_theta = compute_psd(theta_signal, fs)
        _, psd_beta = compute_psd(beta_signal, fs)
        _, psd_alpha = compute_psd(alpha_signal, fs)

        # Calculate mean powers
        mean_power_beta = np.mean(psd_beta)
        mean_power_alpha = np.mean(psd_alpha)

        # Store features
        features[f'PSD_theta_{ch_name}'] = np.sum(psd_theta)
        features[f'PSD_beta_{ch_name}'] = np.sum(psd_beta)
        features[f'PSD_alpha_{ch_name}'] = np.sum(psd_alpha)
        features[f'mean_power_beta_{ch_name}'] = mean_power_beta
        features[f'mean_power_alpha_{ch_name}'] = mean_power_alpha
        features[f'spectral_entropy_beta_{ch_name}'] = entropy(psd_beta)
        features[f'hjorth_activity_{ch_name}'] = np.var(eeg_signal)
        
        # Mobility calculation
        mobility = np.sqrt(np.var(np.diff(eeg_signal)) / np.var(eeg_signal))
        features[f'hjorth_mobility_{ch_name}'] = mobility

    # Asymmetry and ratio calculations
    features['frontal_asymmetry_beta'] = features['PSD_beta_F4'] - features['PSD_beta_F3']
    features['frontal_asymmetry_alpha'] = features['PSD_alpha_F4'] - features['PSD_alpha_F3']
    features['RASM_beta_AF'] = features['PSD_beta_AF4'] / features['PSD_beta_AF3']

    # Theta-Beta Ratio for all nodes
    for ch_name in channels_of_interest:
        features[f'TBR_{ch_name}'] = features[f'PSD_theta_{ch_name}'] / features[f'PSD_beta_{ch_name}']

    return features

def features_to_tensor(feature_dict):
    """Convert feature dictionary to PyTorch tensor"""
    feature_list = []
    for channel in channels_of_interest:
        node_features = [
            feature_dict[f'PSD_theta_{channel}'],
            feature_dict[f'PSD_beta_{channel}'],
            feature_dict[f'mean_power_beta_{channel}'],
            feature_dict[f'mean_power_alpha_{channel}'],
            feature_dict[f'spectral_entropy_beta_{channel}'],
            feature_dict[f'hjorth_activity_{channel}'],
            feature_dict[f'hjorth_mobility_{channel}'],
            feature_dict[f'TBR_{channel}'],
        ]
        feature_list.append(node_features)

    return torch.tensor(feature_list, dtype=torch.float)

def load_and_preprocess_eeg_data(mat_file_path):
    """
    Load EEG data from MATLAB .mat file
    """
    try:
        with h5py.File(mat_file_path, 'r') as file:
            regim_datasub = file['Regim_datasub']
            trial_data = regim_datasub['trial']
            ham_data = regim_datasub['HAM']

            # Extract HAM scores
            ham_scores = ham_data[()].flatten()

            # Determine total number of segments
            num_total_segments = trial_data.shape[1]

            # Preallocate 3D array for EEG segments
            eeg_data = np.zeros((14, 1920, num_total_segments), dtype=np.float32)

            # Populate the eeg_segments array
            for i in range(num_total_segments):
                segment_ref = trial_data[0, i]
                segment = file[segment_ref][()]

                # Ensure correct segment shape
                if segment.shape == (14, 1920):
                    eeg_data[:, :, i] = segment
                elif segment.shape == (1920, 14):
                    eeg_data[:, :, i] = segment.T
                else:
                    print(f"Warning: Unexpected segment shape at index {i}: {segment.shape}")

            return eeg_data, ham_scores

    except Exception as e:
        print(f"Error loading EEG data: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_graph_data(eeg_data, ham_scores):
    """
    Convert EEG data into PyTorch Geometric graph datasets
    """
    graph_dataset = []
    
    # Anxiety category mapping
    def categorize_anxiety(score):
        if score <= 7:
            return 0  # No/Minimal Anxiety
        elif 8 <= score <= 14:
            return 1  # Mild Anxiety
        elif 15 <= score <= 23:
            return 2  # Moderate Anxiety
        else:
            return 3  # Severe Anxiety

    # Process each segment
    for segment_idx in range(eeg_data.shape[2]):
        segment = eeg_data[:, :, segment_idx]
        
        # Extract features for the segment
        features = extract_features(segment)
        
        # Convert features to tensor
        node_features = features_to_tensor(features)
        
        # Compute connectivity matrix (CPCC)
        connectivity_matrix = compute_cpcc_psd_beta(segment)
        
        # Create edge indices
        edge_indices = []
        for i in range(len(channels_of_interest)):
            for j in range(i+1, len(channels_of_interest)):
                # Only add edges above a certain threshold
                if abs(connectivity_matrix[i, j]) > 0.5:
                    edge_indices.extend([i, j])
        
        # Create PyTorch Geometric Data object
        edge_index = torch.tensor(edge_indices, dtype=torch.long).view(2, -1)
        
        # Categorize HAM score
        label = categorize_anxiety(ham_scores[segment_idx])
        
        graph_data = Data(
            x=node_features, 
            edge_index=edge_index, 
            y=torch.tensor(label, dtype=torch.long)
        )
        
        graph_dataset.append(graph_data)
    
    return graph_dataset

class GCN_JK_EEG(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5, num_partitions=2):
        super(GCN_JK_EEG, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_partitions = num_partitions

        assert input_dim % num_partitions == 0, "Input features must be divisible by num_partitions"
        partition_dim = input_dim // num_partitions

        # Create convolutional layers for each partition
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            conv_partitioned = torch.nn.ModuleList()
            for partition in range(num_partitions):
                # First layer: input_dim/num_partitions -> hidden_dim/num_partitions
                if layer == 0:
                    conv_partitioned.append(
                        GCNConv(partition_dim, hidden_dim // num_partitions)
                    )
                # Subsequent layers: hidden_dim/num_partitions -> hidden_dim/num_partitions
                else:
                    conv_partitioned.append(
                        GCNConv(hidden_dim // num_partitions, hidden_dim // num_partitions)
                    )
            self.convs.append(conv_partitioned)
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

        self.jk = JumpingKnowledge(mode='cat')
        self.fc1 = torch.nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # Store outputs from each layer for JK
        layer_outputs = []

        # Process each layer
        for layer_idx in range(self.num_layers):
            # Partition the input features
            partitions = torch.chunk(x, self.num_partitions, dim=-1)

            # Process each partition
            out_parts = []
            for part_idx in range(self.num_partitions):
                part_out = self.convs[layer_idx][part_idx](partitions[part_idx], edge_index)
                out_parts.append(part_out)

            # Concatenate outputs from all partitions
            x = torch.cat(out_parts, dim=-1)

            # Apply batch normalization
            x = self.batch_norms[layer_idx](x)

            # Apply activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Store layer output for JK
            layer_outputs.append(x)

        # Apply JK to combine outputs from all layers
        x = self.jk(layer_outputs)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Final MLP
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)

def prepare_eeg_data(graph_dataset, batch_size=32, train_split=0.7, val_split=0.1):
    """
    Prepare EEG graph dataset for training
    """
    torch.manual_seed(42)
    n = len(graph_dataset)
    train_size = int(train_split * n)
    val_size = int(val_split * n)
    test_size = n - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        graph_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def train_eeg_gnn(model, train_loader, val_loader, device, epochs=200):
    """
    Train the EEG Graph Neural Network with enhanced training strategy
    """
    # More aggressive optimizer with higher learning rate and weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001,  # Slightly increased learning rate
        weight_decay=1e-3  # Increased weight decay for regularization
    )

    # Cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,  # Total number of epochs
        eta_min=1e-5   # Minimum learning rate
    )

    # Training loop with full epoch count
    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index, data.batch)
            
            # Compute loss without label smoothing
            loss = F.nll_loss(out, data.y)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

        # Validation phase
        val_acc = test_eeg_gnn(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step()

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Detailed epoch reporting
        print(f'Epoch {epoch+1:03d}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    # Restore best model state if tracked
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def test_eeg_gnn(model, test_loader, device):
    """
    Test the EEG Graph Neural Network
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total

def main():
    # Path to your .mat file
    mat_file_path = '/content/DASPS+HAM labels.mat'

    # Load EEG data and HAM scores
    eeg_data, ham_scores = load_and_preprocess_eeg_data(mat_file_path)

    # Create graph dataset
    graph_dataset = create_graph_data(eeg_data, ham_scores)
    print(f"Total number of graph samples: {len(graph_dataset)}")
    # Prepare data loaders with stratified sampling
    train_loader, val_loader, test_loader = prepare_eeg_data(
        graph_dataset, 
        batch_size=32,  # You can adjust batch size 
        train_split=0.7, 
        val_split=0.1
    )

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model hyperparameters with potential adjustments
    input_dim = graph_dataset[0].x.shape[1]
    hidden_dim = 128  # Increased from 64
    output_dim = 4  # Number of anxiety categories
    num_layers = 4  # Increased from 3
    num_partitions = 2

    # Initialize model with potential dropout and regularization
    model = GCN_JK_EEG(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.6,  # Slightly increased dropout
        num_partitions=num_partitions
    ).to(device)

    # Train model with increased epochs
    trained_model = train_eeg_gnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=200  # Increased from 100
    )

    # Comprehensive model evaluation
    print("Model Evaluation:")
    train_acc = test_eeg_gnn(trained_model, train_loader, device)
    val_acc = test_eeg_gnn(trained_model, val_loader, device)
    test_acc = test_eeg_gnn(trained_model, test_loader, device)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
