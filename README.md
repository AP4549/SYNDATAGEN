Enhanced Synthetic Data Generator (GAN + VAE)
This project is a web application built using Streamlit that allows users to generate synthetic data using two techniques:

Generative Adversarial Networks (GANs): Used for larger datasets.
Variational Autoencoders (VAEs): Used for smaller datasets (<10MB).
The app is designed to assist in creating synthetic numerical datasets for scenarios where the original dataset is small or unavailable.

Features
Data Upload: Upload any CSV file containing numerical data.
Automatic Technique Selection:
If the dataset is smaller than 10MB, a VAE is used for synthetic data generation.
Otherwise, a GAN is employed.
Synthetic Data Generation:
Generate custom-sized datasets with similar properties to the original dataset.
Automatically handle scaling and normalization.
Performance Monitoring:
View memory and CPU usage during model training.
Visualize training progress through loss curves.
Downloadable Synthetic Data: Download the generated synthetic data as a CSV file.
Tech Stack
Backend Framework: TensorFlow/Keras
Frontend Framework: Streamlit
Languages: Python
Visualization: Matplotlib
Resource Monitoring: Psutil
How It Works
Upload Your Data:
The app accepts .csv files containing numerical data.
Configuration:
Automatically detects dataset size and selects the appropriate model (GAN or VAE).
Adjust hyperparameters like learning rate, epochs, and batch size in the sidebar.
Training:
Train the selected model and monitor discriminator and generator losses (for GAN) or reconstruction loss (for VAE).
Synthetic Data Generation:
Generate synthetic datasets of any size based on the original data distribution.
Download:
Download the generated synthetic dataset in CSV format.
