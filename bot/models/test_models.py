import numpy as np
import matplotlib.pyplot as plt

from layers_blocks import LayersBlock
from tensorflow.keras.layers import Conv2D, Input, Embedding
from tensorflow.keras.layers import Conv1D, Conv1D, Flatten
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model



conv_params ={

        "LayerType": "conv1d",
        "params": {
        "weights_init": {
                "init_type": "random_normal",
                "params": {
                    "mean": 12.1,
                    "stddev": 34.98
                }
            },
            "filters": [32, 64, 128, 128, 128],
            "kernel_size": [5, 5, 5, 5, 5],
            "strides": [1, 1, 1, 1, 1],
            "activation": ["relu", "relu", "relu", "relu", "relu"],
            "padding": ["same", "same", "same", "same", "same"],
        }
}

lstm_params ={

        "LayerType": "lstm",
        "params": {
            "weights_init": {
                "init_type": "random_normal",
                "params": {
                    "mean": 12.1,
                    "stddev": 34.98
                }
            },
            "units": [32],
            "activations": ["linear"],
            "bi": [True],
            "return_sequences": False
        }
}

linear_params = {
            "LayerType": "dense",
            "params": {
                "weights_init": {
                    "init_type": "random_normal",
                    "params": {
                        "mean": 12.1,
                        "stddev": 34.98
                    }
                },
                "units": [128, 64, 3],
                "activations": ["linear", "linear", "linear"]
            }
        }

input_layer = Input(shape=(128, 128))
conv1d_block = LayersBlock(layers_params=conv_params)(input_layer)
flatten_layer = Flatten()(conv1d_block)
linear_block = LayersBlock(layers_params=linear_params)(flatten_layer)
model = Model(inputs=input_layer, outputs=linear_block)
print(model.summary())