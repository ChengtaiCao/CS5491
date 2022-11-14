from sklearn.metrics import confusion_matrix
import pickle
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np



SAVE_PATH = "model.h5"
model = keras.models.load_model(SAVE_PATH)

PATH = "data_dict.pkl"
with open(PATH, "rb") as f:
    data_dict = pickle.load(f)
train_x = data_dict["train_x"]
train_y = data_dict["train_y"]
test_x = data_dict["test_x"]
test_y = data_dict["test_y"]

GENRES_MAP = {
    0: 'blues',
    1: 'classical',
    2: 'country',
    3: 'disco',
    4: 'hiphop',
    5: 'jazz',
    6: 'metal',
    7: 'pop',
    8: 'reggae',
    9: 'rock'
    }

y_prediction = model.predict(test_x)
y_prediction = np.argmax(y_prediction, axis=1)
test_y = np.argmax(test_y, axis=1)
result = confusion_matrix(test_y, y_prediction , normalize='pred')

df_cm = pd.DataFrame(result, index = [GENRES_MAP[i] for i in range(10)],
                  columns = [GENRES_MAP[i]  for i in range(10)])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
