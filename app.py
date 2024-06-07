import numpy as np
from tensorflow.keras.models import load_model
from helpers import extract_features

MODEL = "genreAI_model.keras"
FILE_PATH = "test.wav"
RESULTS = ['blues',
 'classical',
 'country',
 'disco',
 'hiphop',
 'jazz',
 'metal',
 'pop',
 'reggae',
 'rock']

model = load_model(MODEL)

features = extract_features(FILE_PATH)
features = np.expand_dims(features, axis=0)  # Reshape to match model input

# Predict the genre
prediction = model.predict(features)
genre = np.argmax(prediction)

# Handle Result
print(f"Genre is: {RESULTS[genre]}")
