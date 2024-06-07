import numpy as np
from tensorflow.keras.models import load_model
from helpers import extract_features

model = load_model('genreAI_model.keras')

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

features = extract_features(FILE_PATH)
features = np.expand_dims(features, axis=0)  # Reshape to match model input

# Predict the genre
prediction = model.predict(features)
genre = np.argmax(prediction)

# Handle Result
print(f"Genre is: {RESULTS[genre]}")