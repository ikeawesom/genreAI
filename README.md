# genreAI
This project leverages machine learning techniques to classify audio files into different music genres. It uses a neural network trained on extracted audio features such as MFCCs (Mel-Frequency Cepstral Coefficients).

## Features
- Classifies audio files into genres (e.g., rock, pop, classical, jazz).
- Open-source and available for contributions.
- Extracts audio features for analysis.
- Provides a web interface for easy interaction _(coming soon)_.

## Dataset
The dataset used for training the model is the GTZAN dataset, which contains 1000 audio tracks categorized into 10 different genres. The dataset includes extracted audio features (MFCCs) from these audio files. Find out more about GTZAN [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data).

## Model
The model is built using TensorFlow/Keras. It consists of a simple neural network that processes the MFCC features and outputs the predicted genre.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
```sh
git clone https://github.com/ikeawesom/genreAI.git
cd genreAI
```

2. Install the required dependencies:
```sh
pip install -r requirements.txt
```

3. Download the GTZAN dataset:
- From [this](https://github.com/ikeawesom/genreAI/tree/main/datasets) repo OR [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)
- Extract the dataset and place it in the ./datasets directory.
  
4. Train the model:
```sh
python train.py
```

## Usage
1. Modify the `MODEL` and `FILE_PATH` variables within `app.py` to use your preferred model on a sample audio file of your choice. The sample model `genreAI_model.keras` and audio file `test.wav` has been provided within this repo.
```python
MODEL = "genreAI_model.keras"
FILE_PATH = "test.wav"
```

2. Run `app.py`.
```sh
python app.py
>> Genre is rock.
```

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ikeawesom/genreAI/blob/main/LICENSE.md) for details.
