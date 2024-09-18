import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Conv2D, Input, Embedding, Dense
from tensorflow.keras.layers import Conv1D, Conv1D, Flatten, LSTM
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model



input_layer = Input(shape=(None, ))
embedding = Embedding(100, 100)(input_layer)
lstm = LSTM(units=100, return_sequences=True)(embedding)
dense = Dense(units=1000, activation="linear")(lstm)

model = Model(inputs=input_layer, outputs=dense)
print(model.summary())