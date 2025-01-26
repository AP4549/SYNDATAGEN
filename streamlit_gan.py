import os
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from tensorflow.keras import layers, Model
import streamlit as st


# Utility function to monitor system resources
def monitor_resources():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    st.sidebar.write(f"Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")
    st.sidebar.write(f"CPU Usage: {cpu_percent:.2f}%")


# Build GAN
def build_gan(input_dim, latent_dim):
    # Generator
    generator = tf.keras.Sequential([
        layers.Dense(128, activation="relu", input_dim=latent_dim),
        layers.Dense(256, activation="relu"),
        layers.Dense(input_dim, activation="tanh")
    ])
    
    # Discriminator
    discriminator = tf.keras.Sequential([
        layers.Dense(256, activation="relu", input_dim=input_dim),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    return generator, discriminator


# Train GAN
def train_gan(generator, discriminator, data, latent_dim, epochs, batch_size):
    gan = tf.keras.Sequential([generator, discriminator])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    for epoch in range(epochs):
        for _ in range(batch_size):
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_data = generator.predict(noise)
            real_data = data[np.random.randint(0, data.shape[0], batch_size)]
            combined_data = np.vstack((real_data, fake_data))

            labels = np.zeros((2 * batch_size, 1))
            labels[:batch_size] = 1
            
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(combined_data, labels)

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            misleading_labels = np.ones((batch_size, 1))

            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, misleading_labels)
        
        st.write(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")
        monitor_resources()


# Build VAE
def build_vae(input_dim, latent_dim):
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(encoder_inputs)
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = layers.Lambda(lambda p: p[0] + tf.random.normal(tf.shape(p[0])) * tf.exp(p[1] / 2), name="z")([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation="relu")(latent_inputs)
    x = layers.Dense(128, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    outputs = decoder(encoder(encoder_inputs)[2])
    vae = Model(encoder_inputs, outputs, name="vae")

    reconstruction_loss = tf.keras.losses.mean_squared_error(encoder_inputs, outputs)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae.add_loss(reconstruction_loss + kl_loss)
    vae.compile(optimizer="adam")

    return vae, encoder, decoder


# Synthetic Data Generator
def generate_synthetic_data(generator, latent_dim, num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    synthetic_data = generator.predict(noise)
    return pd.DataFrame(synthetic_data)


# Streamlit App
def main():
    st.title("Synthetic Data Generator")
    st.sidebar.header("Configurations")

    model_type = st.sidebar.selectbox("Select Model", ["GAN", "VAE"])
    input_dim = st.sidebar.number_input("Input Dimension", min_value=1, value=10)
    latent_dim = st.sidebar.number_input("Latent Dimension", min_value=1, value=5)
    num_samples = st.sidebar.number_input("Number of Samples", min_value=1, value=1000)
    epochs = st.sidebar.number_input("Epochs", min_value=1, value=100)
    batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=32)

    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")
    if uploaded_file:
        real_data = pd.read_csv(uploaded_file).values
        st.write("Uploaded Data Sample:")
        st.write(pd.DataFrame(real_data).head())
    else:
        st.warning("Please upload a dataset to proceed.")
        return

    if st.sidebar.button("Train Model"):
        if model_type == "GAN":
            generator, discriminator = build_gan(input_dim, latent_dim)
            train_gan(generator, discriminator, real_data, latent_dim, epochs, batch_size)
            synthetic_data = generate_synthetic_data(generator, latent_dim, num_samples)
            st.success("GAN Training Complete!")
        else:
            vae, encoder, decoder = build_vae(input_dim, latent_dim)
            vae.fit(real_data, real_data, epochs=epochs, batch_size=batch_size, verbose=0)
            noise = np.random.normal(0, 1, (num_samples, latent_dim))
            synthetic_data = pd.DataFrame(decoder.predict(noise))
            st.success("VAE Training Complete!")

        st.write("Generated Synthetic Data Sample:")
        st.write(synthetic_data.head())
        csv = synthetic_data.to_csv(index=False)
        st.download_button(label="Download Synthetic Data", data=csv, file_name="synthetic_data.csv", mime="text/csv")

if __name__ == "__main__":
    main()
