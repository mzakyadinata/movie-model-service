import keras
from keras import layers

# Load model lama
old_model = keras.models.load_model(
    "models/encoder_model.keras",
    compile=False
)

print("Model lama berhasil diload")

# Rebuild arsitektur manual
inputs = keras.Input(shape=(384,), name="input")

x = layers.Dense(256, activation="relu", name="dense_1")(inputs)
x = layers.BatchNormalization(name="bn_1")(x)
x = layers.Dropout(0.3, name="dropout_8")(x)

x = layers.Dense(128, activation="relu", name="dense_2")(x)
x = layers.BatchNormalization(name="bn_2")(x)
x = layers.Dropout(0.2, name="dropout_9")(x)

outputs = layers.Dense(64, activation="tanh", name="latent_vector")(x)

new_model = keras.Model(inputs, outputs, name="Encoder")

print("Arsitektur baru berhasil dibuat")

# Copy weights
new_model.set_weights(old_model.get_weights())

print("Weights berhasil dicopy")

# Save model bersih
new_model.save("models/encoder_clean.h5")

print("encoder_clean.h5 berhasil dibuat")