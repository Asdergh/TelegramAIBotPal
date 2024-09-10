import numpy as np
import random as rd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Flatten, Dense
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dropout, LayerNormalization, Reshape
from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras import Model, Sequential
from tensorflow import Module





def _conv1d_block_(params):
        
        layers_n = len(params["filters"])
        layers_list = []

        for layer_n in range(layers_n):
            
            conv_layer = Sequential(layers=[
                Conv1D(filters=params["filters"][layer_n], kernel_size=params["kernel_sizes"][layer_n], strides=params["strides"][layer_n], padding=params["paddings"][layer_n]),
                Activation(activation=params["activations"][layer_n])])
        
        layers_list.append(conv_layer)
        return layers_list


def _conv2d_block_(params):
        
        layers_n = len(params["filters"])
        layers_list = []

        for layer_n in range(layers_n):
            
            conv_layer = Sequential(layers=[
                Conv2D(filters=params["filters"][layer_n], kernel_size=params["kernel_sizes"][layer_n], strides=params["strides"][layer_n], padding=params["paddings"][layer_n]),
                Activation(activation=params["activations"][layer_n])])
            
            layers_list.append(conv_layer)
        
        return layers_list


def _conv3d_block_(params):
        
        layers_n = len(params["filters"])
        layers_list = []

        for layer_n in range(layers_n):
            
            conv_layer = Sequential(layers=[
                Conv3D(filters=params["filters"][layer_n], kernel_size=params["kernel_sizes"][layer_n], strides=params["strides"][layer_n], padding=params["paddings"][layer_n]),
                Activation(activation=params["activations"][layer_n])])
        
            layers_list.append(conv_layer)

        return layers_list

def _conv1d_transpose_block_(params):
    
    layers_n = len(params["filters"])
    layers_list = []

    for layer_n in range(layers_n):
            
        conv_layer = Sequential(layers=[
            Conv1DTranspose(filters=params["filters"][layer_n], kernel_size=params["kernel_sizes"][layer_n], strides=params["strides"][layer_n], padding=params["paddings"][layer_n]),
            Activation(activation=params["activations"][layer_n])])
        
        layers_list.append(conv_layer)

    return layers_list


def _conv2d_transpose_block_(params):
    
    layers_n = len(params["filters"])
    layers_list = []

    for layer_n in range(layers_n):
            
        conv_layer = Sequential(layers=[
            Conv2DTranspose(filters=params["filters"][layer_n], kernel_size=params["kernel_sizes"][layer_n], strides=params["strides"][layer_n], padding=params["paddings"][layer_n]),
            Activation(activation=params["activations"][layer_n])])
        
        layers_list.append(conv_layer)

    return layers_list


def _conv3d_transpose_block_(params):
    
    layers_n = len(params["filters"])
    layers_list = []
    for layer_n in range(layers_n):
            
        conv_layer = Sequential(layers=[
            Conv3DTranspose(filters=params["filters"][layer_n], kernel_size=params["kernel_sizes"][layer_n], strides=params["strides"][layer_n], padding=params["paddings"][layer_n]),
            Activation(activation=params["activations"][layer_n])])
        
        layers_list.append(conv_layer)

    return layers_list


def _lstm_block_(params):

    layers_n = len(params["units"])
    layers_list = []
    for layer_n in range(layers_n - 1):
        
        if params["bi"][layer_n]:
            lstm_layer = Bidirectional(LSTM(units=params["units"][layer_n], return_sequences=True))
        
        else:
            lstm_layer = LSTM(units=params["units"][layer_n], return_sequences=True)

        layer = Sequential(layers=[
            lstm_layer,
            Activation(params["activations"][layer_n])])
        layers_list.append(layer)
    
    output_layer = LSTM(units=params["units"][-1])
    layers_list.append(output_layer)
    
    return layers_list



    

    


class LayersBlock(Module):

    def __init__(self, layers_params, name=None, 
                normalization=True, dropout=True,
                dropout_rate=0.56, epsilon=0.01):
        
        super().__init__(name)
        self.layers_params = layers_params
        self.__block_generation__ = {
            "conv1d": _conv1d_block_,
            "conv2d": _conv2d_block_,
            "conv3d": _conv3d_block_,
            "conv1d_transpose": _conv1d_transpose_block_,
            "conv2d_transpose": _conv2d_transpose_block_,
            "conv3d_transpose": _conv3d_transpose_block_,
            "lstm": _lstm_block_
        }

        layer_type = self.layers_params["LayerType"]
        params = self.layers_params["params"]
        self.layers_list = self.__block_generation__[layer_type](params=params)
        
        if dropout:
            
            layer = Dropout(rate=dropout_rate)
            self.layers_list.append(layer)
        
        if normalization:
             
            layer = LayerNormalization(epsilon=epsilon)
            self.layers_list.append(layer)
        
    def __call__(self, x):
         
        for layer in self.layers_list:
            x = layer(x)
        
        return x
    
        












    
    
    

    
