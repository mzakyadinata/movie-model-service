import keras

model = keras.models.load_model(
    "models/encoder_model.keras",
    compile=False
)

model.save("models/encoder_model.h5")