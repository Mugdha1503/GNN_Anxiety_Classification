Project Overview

This project explores the use of Graph Neural Networks (GNNs) to classify human anxiety levels based on EEG signals. By representing EEG channels as graph nodes and their connections as edges, the GNN is designed to capture both spatial and functional relationships between different brain regions. The project also focuses on optimizing memory usage in GNNs to handle large-scale data and avoid out-of-memory (OOM) issues.

Key Objectives

Classify human anxiety states into four categories: Normal, Light, Moderate, and Severe, using EEG data processed through GNNs.
Optimize memory usage in GNNs through sampling techniques and WholeGraph, enhancing scalability and performance in large-scale graphs.
Explore the use of EEG signals in a graph-based model to improve classification accuracy while reducing the computational overhead associated with large datasets.

Methodology

Data Collection: EEG signals are collected and represented as graphs, with nodes representing EEG channels and edges representing connections (based on physical proximity or functional connectivity).

Feature Extraction: Extract key features from EEG signals, such as Mean Power, Rational Asymmetry, and Asymmetry Index.

GNN Model: Implement Graph Convolutional Networks (GCNs) with pooling layers to aggregate node-level features and classify anxiety states.

Optimization: Memory optimization techniques like node-wise sampling and WholeGraph are employed to improve model efficiency and scalability.

Classification: The GNN model outputs one of four anxiety levels based on EEG data.

Key Features:


Graph Representation of EEG Data: EEG channels are transformed into a graph structure, allowing for better pattern recognition.

Graph Convolution Layers: Graph convolutions aggregate features from neighboring nodes, capturing spatial and functional brain activity.

Memory Optimization: Techniques such as node sampling and WholeGraph improve memory management and scalability.

EEG Feature Analysis: Frequency-based features (e.g., delta, theta, alpha, beta, gamma bands) are used for accurate classification.

Tools and Technologies:


Python

PyTorch for GNN implementation

WholeGraph for memory optimization

Keras for model training and evaluation


Future Work

The next steps involve further optimizing GNN models for deployment on low-power edge devices, enabling real-time anxiety classification. Additionally, we plan to enhance the scalability of the model for larger datasets and explore other mental health-related applications using EEG data.
