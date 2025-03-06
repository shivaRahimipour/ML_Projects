############################ Implementing a Basic Autoencoder

## Step 1:Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import mnist


## Step 2:Load and Preprocess Data
# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values (0 to 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images to 784-dimensional vectors
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))


## Step 3:Build the Autoencoder
# Define input layer
input_dim = 784  # 28x28 images flattened
encoding_dim = 32  # Size of the compressed representation

# Encoder
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_img, decoded)

# Compile model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

## Step 4: Train the Autoencoder
autoencoder.fit(
    x_train, x_train,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

## Step 5: Test and Visualize Results
# Encode and decode test images
decoded_imgs = autoencoder.predict(x_test)

# Reshape for visualization
x_test_reshaped = x_test.reshape(-1, 28, 28)
decoded_imgs_reshaped = decoded_imgs.reshape(-1, 28, 28)

# Display original and reconstructed images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))

for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_reshaped[i], cmap="gray")
    plt.axis("off")

    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_reshaped[i], cmap="gray")
    plt.axis("off")

plt.show(block=False)


##################### Denoising Autoencoder (DAE) using TensorFlow/Keras
#reconstruct data from a corrupted (noisy) version of the input. It helps improve robustness and feature learning.
from tensorflow.keras.layers import GaussianNoise

#Step 1: Add Gaussian Noise
noise_factor = 0.5  # Adjust noise level

# Create noisy images
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip values between 0 and 1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#Step 2: Build the Denoising Autoencoder
# Define input layer
input_dim = 784  # 28x28 images flattened
encoding_dim = 64  # Size of the compressed representation

# Encoder with Gaussian Noise layer
input_img = Input(shape=(input_dim,))
noisy_input = GaussianNoise(0.2)(input_img)  # Add noise layer
encoded = Dense(encoding_dim, activation='relu')(noisy_input)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_img, decoded)

# Compile model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#Step 3: Train the Autoencoder
autoencoder.fit(
    x_train_noisy, x_train,  # Train on noisy input but predict clean images
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

#Step 4: Test and Visualize Results
# Encode and decode test images
decoded_imgs = autoencoder.predict(x_test_noisy)

# Reshape for visualization
x_test_reshaped = x_test.reshape(-1, 28, 28)
x_test_noisy_reshaped = x_test_noisy.reshape(-1, 28, 28)
decoded_imgs_reshaped = decoded_imgs.reshape(-1, 28, 28)

# Display original, noisy, and denoised images
n = 10  # Number of images to display
plt.figure(figsize=(20, 6))

for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_reshaped[i], cmap="gray")
    plt.axis("off")
    if i == 0:
        ax.set_title("Original")

    # Noisy images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy_reshaped[i], cmap="gray")
    plt.axis("off")
    if i == 0:
        ax.set_title("Noisy")

    # Denoised images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs_reshaped[i], cmap="gray")
    plt.axis("off")
    if i == 0:
        ax.set_title("Denoised")

plt.show()