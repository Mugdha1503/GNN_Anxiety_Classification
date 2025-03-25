import numpy as np
import scipy.signal as signal
from scipy.stats import entropy
from sklearn.preprocessing import normalize
import h5py
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx

# Define frequency bands
theta_band = (4, 7)  # Theta band: 4–7 Hz
beta_band = (13, 30) # Beta band: 13–30 Hz
alpha_band = (8, 12) # Alpha band: 8–12 Hz

# List of channels to extract
channels_of_interest = ['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'P7', 'P8']

# Load the dataset (example for one participant's .mat file)
with h5py.File('S01preprocessed.mat', 'r') as file:
    data = file['data']  # Shape: (14, 1920, 6)
    eeg_data = np.array(data)

# Map channel indices for the 8 channels
channel_indices = {'AF3': 0, 'AF4': 1, 'F3': 2, 'F4': 3, 'FC5': 4, 'FC6': 5, 'P7': 10, 'P8': 11}

# Helper function to apply bandpass filter
def bandpass_filter(eeg_signal, fs, band):
    nyquist = 0.5 * fs
    low, high = band
    low /= nyquist
    high /= nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.lfilter(b, a, eeg_signal)

# Helper function to compute PSD
def compute_psd(eeg_signal, fs):
    freqs, psd = signal.welch(eeg_signal, fs, nperseg=256)
    return freqs, psd

# Function to compute features for one segment
def extract_features(segment, fs=128):
    features = {}

    # For each selected channel
    for ch_name, ch_index in channel_indices.items():
        eeg_signal = segment[ch_index, :]

        # Apply bandpass filter for theta, beta, and alpha
        theta_signal = bandpass_filter(eeg_signal, fs, theta_band)
        beta_signal = bandpass_filter(eeg_signal, fs, beta_band)
        alpha_signal = bandpass_filter(eeg_signal, fs, alpha_band)

        # Compute PSD for theta, beta, and alpha
        _, psd_theta = compute_psd(theta_signal, fs)
        _, psd_beta = compute_psd(beta_signal, fs)
        _, psd_alpha = compute_psd(alpha_signal, fs)

        # Calculate mean power for beta and alpha bands
        mean_power_beta = np.mean(psd_beta)
        mean_power_alpha = np.mean(psd_alpha)

        # Store features
        features[f'PSD_theta_{ch_name}'] = np.sum(psd_theta)
        features[f'PSD_beta_{ch_name}'] = np.sum(psd_beta)
        features[f'PSD_alpha_{ch_name}'] = np.sum(psd_alpha)
        features[f'mean_power_beta_{ch_name}'] = mean_power_beta
        features[f'mean_power_alpha_{ch_name}'] = mean_power_alpha

        # Spectral Entropy in beta band
        features[f'spectral_entropy_beta_{ch_name}'] = entropy(psd_beta)

        # Hjorth parameters
        features[f'hjorth_activity_{ch_name}'] = np.var(eeg_signal)
        mobility = np.sqrt(np.var(np.diff(eeg_signal)) / np.var(eeg_signal))
        features[f'hjorth_mobility_{ch_name}'] = mobility

    # Frontal asymmetry (F3 and F4)
    features['frontal_asymmetry_beta'] = features['PSD_beta_F4'] - features['PSD_beta_F3']
    features['frontal_asymmetry_alpha'] = features['PSD_alpha_F4'] - features['PSD_alpha_F3']

    # Rational Asymmetry (RASM) in Beta band for AF3 and AF4
    features['RASM_beta_AF'] = features['PSD_beta_AF4'] / features['PSD_beta_AF3']

    # Theta-Beta Ratio (TBR) for all nodes
    for ch_name in channel_indices.keys():
        features[f'TBR_{ch_name}'] = features[f'PSD_theta_{ch_name}'] / features[f'PSD_beta_{ch_name}']

    return features

# Function to compute complex Pearson correlation coefficient
def complex_pearson_correlation(signal_x, signal_y):
    """
    Compute the complex Pearson correlation coefficient between two signals.
    
    Parameters:
    - signal_x: First time series (real-valued)
    - signal_y: Second time series (real-valued)
    
    Returns:
    - complex_corr: Complex correlation coefficient
    """
    # Convert signals to analytic signals (complex-valued) using Hilbert transform
    analytic_x = signal.hilbert(signal_x)
    analytic_y = signal.hilbert(signal_y)
    
    # Calculate means
    mean_x = np.mean(analytic_x)
    mean_y = np.mean(analytic_y)
    
    # Calculate numerator (covariance)
    numerator = np.sum((analytic_x - mean_x) * np.conjugate(analytic_y - mean_y))
    
    # Calculate denominator (product of standard deviations)
    std_x = np.sqrt(np.sum(np.abs(analytic_x - mean_x) ** 2))
    std_y = np.sqrt(np.sum(np.abs(analytic_y - mean_y) ** 2))
    
    # Calculate complex correlation coefficient
    complex_corr = numerator / (std_x * std_y)
    
    return complex_corr

# Function to compute complex Pearson correlation and construct a complete graph
def compute_complex_pearson_correlation(segment):
    # Only consider the 8 channels of interest (subset)
    selected_segment = segment[list(channel_indices.values()), :]  # Filter only the selected 8 channels

    num_channels = selected_segment.shape[0]
    complex_pearson_matrix = np.zeros((num_channels, num_channels), dtype=complex)

    # Compute complex Pearson correlation for each pair of selected channels
    for i in range(num_channels):
        for j in range(num_channels):
            complex_pearson_matrix[i, j] = complex_pearson_correlation(selected_segment[i, :], selected_segment[j, :])

    return complex_pearson_matrix

# Extract features and construct edges for all segments
all_features = []
all_edge_weights = []
for segment_idx in range(5):
    segment = eeg_data[:, :, segment_idx]  # One segment of data

    # Extract node features
    features = extract_features(segment)
    all_features.append(features)

    # Compute edges using complex Pearson correlation
    complex_pearson_matrix = compute_complex_pearson_correlation(segment)
    edges = []
    for i in range(complex_pearson_matrix.shape[0]):
        for j in range(complex_pearson_matrix.shape[1]):
            # Store the complex correlation coefficient as magnitude and phase
            magnitude = np.abs(complex_pearson_matrix[i, j])
            phase = np.angle(complex_pearson_matrix[i, j])
            edges.append((i, j, magnitude, phase))  # Store the complete edge with magnitude and phase

    all_edge_weights.append(edges)

for segment_idx in range(5, 11):
    segment = eeg_data[:, :, segment_idx]  # One segment of data

    # Extract node features
    features = extract_features(segment)
    all_features.append(features)

    # Compute edges using complex Pearson correlation
    complex_pearson_matrix = compute_complex_pearson_correlation(segment)
    edges = []
    for i in range(complex_pearson_matrix.shape[0]):
        for j in range(complex_pearson_matrix.shape[1]):
            # Store the complex correlation coefficient as magnitude and phase
            magnitude = np.abs(complex_pearson_matrix[i, j])
            phase = np.angle(complex_pearson_matrix[i, j])
            edges.append((i, j, magnitude, phase))  # Store the complete edge with magnitude and phase

    all_edge_weights.append(edges)

# Function to convert node features dictionary into tensor
def features_to_tensor(feature_dict):
    feature_list = []
    for channel in channels_of_interest:
        # Extracting the features related to each channel (assuming sorted order of channels)
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

    # Convert the list of features to a PyTorch tensor
    return torch.tensor(feature_list, dtype=torch.float)

# Construct the graph for each segment
def create_graph(segment_index):
    # Get node features and edges for the given segment
    segment_features = all_features[segment_index]
    segment_edges = all_edge_weights[segment_index]

    # Convert node features to tensor (shape: [num_nodes, num_features])
    x = features_to_tensor(segment_features)

    # Extract edge indices and edge weights
    edge_index = []  # Edge indices (source, target) pairs
    edge_attr = []   # Edge attributes (magnitude and phase)

    for (i, j, magnitude, phase) in segment_edges:
        edge_index.append([i, j])  # Adding source-target pair
        edge_attr.append([magnitude, phase])  # Adding both magnitude and phase as edge attributes

    # Convert edge_index to a PyTorch tensor of shape [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Convert edge_attr to a PyTorch tensor of shape [num_edges, 2]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create the graph using PyTorch Geometric's Data structure
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph, segment_edges

# Updated function to visualize the graph with complex correlation values
def visualize_eeg_graph_complex_edges(segment_index):
    """
    Visualize the EEG graph for a specific segment with complex correlation edges.
    
    Parameters:
    - segment_index: Index of the segment to visualize
    """
    # Create PyTorch Geometric graph and get edge data
    graph, edges = create_graph(segment_index)
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes with channel names as labels
    for i, channel in enumerate(channels_of_interest):
        G.add_node(i, label=channel)
    
    # Add all edges from complex correlation matrix except self-loops
    for i, j, magnitude, phase in edges:
        if i != j:  # Exclude self-loops
            G.add_edge(i, j, weight=magnitude, phase=phase)
    
    # Get position layout - using circular layout for EEG channels
    pos = nx.circular_layout(G)
    
    # Create a color map for edges based on phase values
    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        # Use HSV color space: Hue based on phase, Saturation=1, Value based on magnitude
        # Map phase from [-π, π] to [0, 1] for hue
        hue = (data['phase'] + np.pi) / (2 * np.pi)
        # Map color from HSV to RGB
        edge_colors.append(plt.cm.hsv(hue))
        
        # Width proportional to magnitude (with minimum width)
        edge_widths.append(max(0.5, data['weight'] * 3))
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
    
    # Draw edges with colors and widths based on weights
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=15, font_weight='bold')
    
    # Add edge weight labels (showing magnitude and phase)
    edge_labels = {(u, v): f"{d['weight']:.2f}∠{d['phase']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Set title and remove axis
    plt.title(f"EEG Channel Connectivity - Complex Correlation (Segment {segment_index})", fontsize=16)
    plt.axis('off')
    
    # Add a colorbar for phase values
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6)
    cbar.set_label('Phase (radians)', fontsize=12)
    
    # Add a legend for magnitude
    plt.text(1.1, 0.5, "Edge Width:", fontsize=12, transform=plt.gca().transAxes)
    plt.text(1.1, 0.45, "Proportional to Magnitude", fontsize=10, transform=plt.gca().transAxes)
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Function to compare multiple segments with complex correlation
def compare_segments_complex(segment_indices):
    """
    Compare multiple segments by visualizing their graphs side by side with complex correlation.
    
    Parameters:
    - segment_indices: List of segment indices to compare
    """
    n_segments = len(segment_indices)
    fig = plt.figure(figsize=(6*n_segments, 12))
    
    for i, seg_idx in enumerate(segment_indices):
        # Create subplot
        plt.subplot(1, n_segments, i+1)
        
        # Get graph data
        _, edges = create_graph(seg_idx)
        
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes with channel names as labels
        for j, channel in enumerate(channels_of_interest):
            G.add_node(j, label=channel)
        
        # Add all edges except self-loops
        for src, tgt, magnitude, phase in edges:
            if src != tgt:  # Exclude self-loops
                G.add_edge(src, tgt, weight=magnitude, phase=phase)
        
        # Get position layout - using circular layout for EEG channels
        pos = nx.circular_layout(G)
        
        # Create a color map for edges based on phase values
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            # Use HSV color space: Hue based on phase
            hue = (data['phase'] + np.pi) / (2 * np.pi)
            edge_colors.append(plt.cm.hsv(hue))
            # Width proportional to magnitude
            edge_widths.append(max(0.5, data['weight'] * 3))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue', alpha=0.8)
        
        # Draw edges with colors and widths based on weights
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Set title and remove axis
        plt.title(f"Segment {seg_idx} - Complex Correlation", fontsize=14)
        plt.axis('off')
    
    # Add a common title
    plt.suptitle("Comparison of EEG Connectivity Across Segments (Complex Correlation)", fontsize=16)
    
    # Add a common colorbar for phase values
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Phase (radians)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust for suptitle and colorbar
    plt.show()

# Print all complex correlation values for a segment
def print_complex_pearson_values(segment_index):
    """
    Print all complex Pearson correlation values for a specific segment.
    
    Parameters:
    - segment_index: Index of the segment
    """
    segment = eeg_data[:, :, segment_index]
    complex_pearson_matrix = compute_complex_pearson_correlation(segment)
    
    print(f"Complex Pearson Correlation Matrix for Segment {segment_index}:")
    print("-----------------------------------------------------")
    print(f"{'Channel':<8}", end="")
    for ch in channels_of_interest:
        print(f"{ch:<12}", end="")
    print()
    
    for i, ch1 in enumerate(channels_of_interest):
        print(f"{ch1:<8}", end="")
        for j, ch2 in enumerate(channels_of_interest):
            # Print magnitude and phase in polar form
            magnitude = np.abs(complex_pearson_matrix[i, j])
            phase = np.angle(complex_pearson_matrix[i, j])
            print(f"{magnitude:.2f}∠{phase:.2f}  ", end="")
        print()
    print("-----------------------------------------------------")

# Example usage:
segment_index = 0  # Example: visualize graph for the first segment

# Visualize the graph with complex correlation edges
visualize_eeg_graph_complex_edges(segment_index)

# Print all complex correlation values
print_complex_pearson_values(segment_index)

# Compare multiple segments example with complex correlation
compare_segments_complex([0, 1, 2])  # Compare first three segments
