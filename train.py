import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('./datasets/sample-3-sec.csv')

# get labels from the dataset
label_arr = data["label"].unique().tolist()
label_obj = {k: v for v, k in enumerate(label_arr)}

# replace labels with integers
data['label'] = data['label'].map(label_obj).fillna(data['label'])

# trim columns
trim = data.drop(["filename","length"], axis=1)
trim = trim.dropna()

# start training process
test_size = 0.2
seed = 42
X = trim.drop('label', axis=1)
y = trim['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# using standardscaler class
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

i = 30
n = X_train.shape[1]
o = len(label_arr)
model = Sequential([
    Input(shape=(n,)),
    Dense(n*3, activation='relu'),
    Dense(n*2, activation='relu'),
    Dense(n*1, activation='relu'),
    Dense(o, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=i, validation_data=(X_test, y_test))

# saves model
model.save('audio-to-genre-model.keras')
print("Model saved.")