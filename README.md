# genreAI
This project leverages machine learning techniques to classify audio files into different music genres. It uses a neural network trained on extracted audio features such as MFCCs (Mel-Frequency Cepstral Coefficients).

## Features
- Classifies audio files into genres (e.g., rock, pop, classical, jazz).
- Open-source and available for contributions.

## Dataset
The dataset used for training the model is the GTZAN dataset, which contains 1000 audio tracks categorized into 10 different genres. The dataset includes extracted audio features (MFCCs) from these audio files. Find out more about GTZAN [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data).

## Model
The model is built using TensorFlow/Keras. It consists of a simple neural network that processes the MFCC features and outputs the predicted genre.

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ikeawesom/genreAI/blob/main/LICENSE.md) for details.
