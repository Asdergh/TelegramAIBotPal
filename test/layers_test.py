import numpy as np
import matplotlib.pyplot as plt
import json as js
import os
import tensorflow.keras.backend as K


from tensorflow.keras.layers import Input, Dense, Activation, Reshape, LayerNormalization, BatchNormalization, Add, Conv3D, Conv3DTranspose, MaxPool2D
from tensorflow.keras.layers import LSTM, GRU, Masking, Bidirectional, Dropout, Conv2D, Conv2DTranspose, Flatten, Conv1D, MaxPool1D
from tensorflow.keras.layers import Concatenate, Lambda, Embedding, Multiply, Layer, Conv1D, Attention, RepeatVector, Conv1DTranspose, MaxPool3D
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.random import categorical
from tensorflow import map_fn, expand_dims, convert_to_tensor
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import RandomNormal, RandomUniform





test_params = {
    "conv2d": {
        "layer_type": "conv2d",
        "layers": [
            [32, {"kernel_size": 3, "strides": 2}  ],
            [32, {"kernel_size": 3, "strides": 2}  ],
            [128, {"kernel_size": 3, "strides": 2} ],
            [128, {"kernel_size": 3, "strides": 2} ]
        ],

        "weigths_init": {
            "type": "random_normal",
            "params": {
                "mean": 0.12,
                "stddev": 0.34
            }
        }
    },

    "linear": {
        "layers": [
            [32, {"activation": "relu"}  ],
            [32, {"activation": "relu"}  ],
            [128, {"activation": "relu"} ],
            [128, {"activation": "relu"} ],
            [128, {"activation": "relu"} ]
        ],

        "weigths_init": {
            "type": "random_normal",
            "params": {
                "mean": 0.12,
                "stddev": 0.34
            }
        }
    }
}




layers_collection = {
    "conv1d": Conv2D,
    "conv2d": Conv2D,
    "conv3d": Conv2D,
    "LSTM": LSTM,
    "linear": Dense
}

input_shape = (456, 456, 3)
input_layer = Input(shape=input_shape)
conv_layer = input_layer

weigths_init = None
if "weiths_init" in test_params["conv2d"].keys():
    weigths_init = RandomNormal(mean=test_params["conv2d"]["weigths_init"]["params"]["mean"], stddev=test_params["conv2d"]["weigths_init"]["params"]["stddev"])

for params in test_params["conv2d"]["layers"]:
    conv_layer = layers_collection["conv2d"](filters=params[0], kernel_initializer=weigths_init, **params[1])(conv_layer)

dense_layer = Flatten()(conv_layer)
for params in test_params["linear"]["layers"]:
    dense_layer = layers_collection["linear"](units=params[0], kernel_initializer=weigths_init, **params[1])(dense_layer)

model = Model(inputs=input_layer, outputs=dense_layer)
print(model.summary())

