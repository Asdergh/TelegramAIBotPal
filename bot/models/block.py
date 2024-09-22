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
from tensorflow import Module




class Block(Module):

    def __init__(self, block_params, name=None,
                norm_each_layer=False, normalization=False, 
                dropout=False, dropout_rate=0.56, 
                epsilon=0.01):
        
        super().__init__(name)
        self.block_params = block_params 
        self.layers_list = []       
        
        self.norm_params = None
        if norm_each_layer:
            self.norm_param = [dropout_rate, epsilon]

        self.__layer_generation__ = {
            "conv1d": Conv1D,
            "conv2d": Conv2D,
            "conv3d": Conv3D,
            "conv1d_transpose": Conv1DTranspose,
            "conv2d_transpose": Conv2DTranspose,
            "conv3d_transpose": Conv3DTranspose,
            "lstm": LSTM,
            "dense": Dense
        }

        self.__layers_block__()        
        if dropout:
            
            layer = Dropout(rate=dropout_rate)
            self.layers_list.append(layer)
        
        if normalization:
             
            layer = LayerNormalization(epsilon=epsilon)
            self.layers_list.append(layer)
    
    
        
    def __layers_block__(self):
        
        
        layer_type = self.block_params["layers_type"]
        weights_initializer = None
        if "weigths_init" in self.block_params.keys():
            weights_initializer = self.__get_weights_initializer__(weights_params=self.block_params["weights_init"])

        for layer_p in self.block_params:

            layer = self.__layer_generation__[layer_type](layer_p[0], kernel_initializer=weights_initializer)
            self.layers_list.append(layer)
        
    def __get_weights_initializer__(self, weights_params):
        
        if "weights_init" in self.block_params.keys():

            init_type = weights_params["init_type"]
            params = weights_params["params"]

            if init_type == "random_normal":
                weights_init = RandomNormal(mean=params["mean"], stddev=params["stddev"])
            
            elif init_type == "random_uniform":
                weights_init = RandomUniform(minval=params["min_val"], maxval=params["max_val"])
            
            return weights_init

    def __call__(self, x):
         
        for layer in self.layers_list:
            x = layer(x)
        
        return x
    
    

        











    
    
    

    
