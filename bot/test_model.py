import numpy as np
import matplotlib.pyplot as plt
import random as rd
import json as js
import os

from models import RnnConv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical





images_data = np.random.normal(0, 1, (100, 224, 224, 3))
sequences_data = np.random.randint(0, 1000, size=(100, 100))
total_words = np.max(sequences_data)
tokenizer = Tokenizer()


params_json = {
    "input_shape": [224, 224, 3],
    "total_labels_n": int(total_words) + 1,
  
    "weights_init": {
        "mean": 0.0,
        "stddev": 1.0
    },
    
    "encoder_params": {
        "single_dropout":   False,
        "filters":          [128, 64, 64, 32],
        "kernel_size":      [3, 3, 3, 3],
        "strides":          [1, 2, 2, 1],
        "dropout_rates":    [0.26, 0.26, 0.26, 0.26],
        "activations":      ["linear", "linear", "linear", "linear"],
        "output_activation": "tanh"
    },

    "decoder_params": {
        "embedding_dim":    100,
        "single_dropout":   False,
        "units":            [256, 256, 256, 256],
        "bidirectional":    [0, 0, 1, 1],
        "dropout_rates":    [0.26, 0.26, 0.26, 0.26]
    }
}


print(images_data.shape, sequences_data.shape)

rnn_conv = RnnConv(params_json=params_json)
print(rnn_conv.encoder.summary())
print(rnn_conv.decoder.summary())
print(rnn_conv.model.summary())
rnn_conv.train_model(train_images=images_data, train_sequences=sequences_data, epochs=10, batch_size=32)

