import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
import os
import psutil
import matplotlib.pyplot as plt

st.title("Enhanced GAN-Based Synthetic Data Generator")
st.write("This app demonstrates how to build a Generative Adversarial Network (GAN) to generate synthetic data.")

def load_data(file_buffer):
    try:
        data = pd.read_csv(file_buffer)
        st.success("Data loaded successfully.")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def build_gan(input_dim, output_dim, learning_rate):
    generator = Sequential(
        [
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dense(output_dim, activation='linear')
        ],
        name="generator"
    )

    discriminator = Sequential(
        [
            Dense(512, input_dim=output_dim),
            LeakyReLU(alpha=0.2),
            Dense(256),
            LeakyReLU(alpha=0.2),
            Dense(1, activation='sigmoid')
        ],
        name="discriminator"
    )

    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy']
    )
    discriminator.trainable = False

    gan = Sequential([generator, discriminator], name="gan")
    gan.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate)
    )

    return gan, generator, discriminator

def train_gan(gan, generator, discriminator, real_data, epochs, batch_size):
    n_samples = real_data.shape[0]
    input_dim = generator.input_shape[1]
    n_batches = max(1, n_samples // batch_size)

    d_losses, g_losses = [], []

    for epoch in range(epochs):
        d_loss_epoch, g_loss_epoch = 0, 0

        for batch in range(n_batches):
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent(interval=0.1)
            st.write(f"Memory Usage: {memory_usage}% | CPU Usage: {cpu_usage}%")

            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            real_samples = real_data.iloc[start_idx:end_idx].values

            noise = np.random.normal(0, 1, (real_samples.shape[0], input_dim))
            fake_samples = generator.predict(noise)

            combined_samples = np.vstack([real_samples, fake_samples])
            real_labels = np.ones(real_samples.shape[0]) * np.random.uniform(0.8, 1.0)
            fake_labels = np.zeros(fake_samples.shape[0]) * np.random.uniform(0.0, 0.2)
            labels = np.hstack([real_labels, fake_labels])

            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(combined_samples, labels)
            d_loss_epoch += d_loss[0] if isinstance(d_loss, (list, tuple)) else d_loss

            noise = np.random.normal(0, 1, (real_samples.shape[0], input_dim))
            misleading_labels = np.ones(real_samples.shape[0])
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, misleading_labels)
            g_loss_epoch += g_loss[0] if isinstance(g_loss, (list, tuple)) else g_loss

        avg_d_loss = d_loss_epoch / n_batches
        avg_g_loss = g_loss_epoch / n_batches
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        st.write(f"Epoch {epoch + 1}/{epochs} | Discriminator Loss: {avg_d_loss:.4f} | Generator Loss: {avg_g_loss:.4f}")

    st.success("GAN training complete!")
    return d_losses, g_losses

def generate_synthetic_data(generator, num_samples, original_data):
    noise = np.random.normal(0, 1, (num_samples, generator.input_shape[1]))
    synthetic_data = generator.predict(noise)

    for i, column in enumerate(original_data.columns):
        col_min, col_max = original_data[column].min(), original_data[column].max()

        if np.issubdtype(original_data[column].dtype, np.integer):
            synthetic_data[:, i] = np.round(synthetic_data[:, i])

        synthetic_data[:, i] = np.clip(synthetic_data[:, i], col_min, col_max)

        unique_mod10 = np.unique(original_data[column] % 10)
        if len(unique_mod10) == 1 and unique_mod10[0] == 0:
            synthetic_data[:, i] = np.round(synthetic_data[:, i] / 10) * 10

    synthetic_df = pd.DataFrame(synthetic_data, columns=original_data.columns)

    for col in original_data.columns:
        if np.issubdtype(original_data[col].dtype, np.integer):
            synthetic_df[col] = synthetic_df[col].astype(int)

    return synthetic_df

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], accept_multiple_files=False, key="file_uploader")
if uploaded_file:
    real_data = load_data(uploaded_file)

    if real_data is not None:
        st.write("Data Preview:")
        st.dataframe(real_data.head())

        numeric_data = real_data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            st.error("No numeric columns found in the dataset.")
        else:
            st.write("Numeric Data Preview:")
            st.dataframe(numeric_data.head())

            if numeric_data.nunique().min() < 2:
                st.warning("Warning: Some columns have very low diversity. Consider using a more diverse dataset.")

            st.sidebar.header("GAN Configuration")
            epochs = min(max(10, numeric_data.shape[0] // 100), 200)
            batch_size = min(max(32, numeric_data.shape[0] // 50), 256)
            learning_rate = st.sidebar.slider("Learning Rate:", min_value=0.00001, max_value=0.01, value=0.0002, step=0.00001)

            st.sidebar.write(f"Calculated Epochs: {epochs}")
            st.sidebar.write(f"Calculated Batch Size: {batch_size}")

            gan, generator, discriminator = build_gan(
                input_dim=numeric_data.shape[1],
                output_dim=numeric_data.shape[1],
                learning_rate=learning_rate
            )

            if st.button("Train GAN"):
                with st.spinner("Training..."):
                    d_losses, g_losses = train_gan(gan, generator, discriminator, numeric_data, epochs=epochs, batch_size=batch_size)

                st.subheader("Loss Curves")
                fig, ax = plt.subplots()
                ax.plot(d_losses, label='Discriminator Loss')
                ax.plot(g_losses, label='Generator Loss')
                ax.set_title('Loss Curves')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)

            num_samples = st.sidebar.number_input("Enter the number of synthetic rows to generate:", min_value=1, value=10, step=1)
            if st.button("Generate Synthetic Data"):
                synthetic_df = generate_synthetic_data(generator, num_samples, numeric_data)

                st.write("Generated Synthetic Data:")
                st.dataframe(synthetic_df)

                csv = synthetic_df.to_csv(index=False)
                st.download_button(
                    label="Download Synthetic Data",
                    data=csv,
                    file_name="synthetic_data.csv",
                    mime="text/csv"
                )
