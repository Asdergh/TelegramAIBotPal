import numpy as np
import random as rd
import matplotlib.pyplot as plt
import tensorflow as tf
import layers as ly

from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Flatten, Dense
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dropout, LayerNormalization, Reshape
from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Zeros
from tensorflow.keras import Model, Sequential
from tensorflow import Module




class LayersBlock(Module):

    def __init__(self, layers_params, name=None,
                norm_each_layer=False, normalization=False, 
                dropout=False, dropout_rate=0.56, 
                epsilon=0.01):
        
        super().__init__(name)
        self.layers_params = layers_params        
        
        self.norm_params = None
        if norm_each_layer:
            self.norm_param = [dropout_rate, epsilon]

        self.__layer_generation__ = {
            "conv1d": getattr(ly, "__conv1d_layer__"),
            "conv2d": getattr(ly, "__conv2d_layer__"),
            "conv3d": getattr(ly, "__conv3d_layer__"),
            "conv1d_transpose": getattr(ly, "__conv1d_transpose_layer__"),
            "conv2d_transpose": getattr(ly, "__conv2d_transpose_layer__"),
            "conv3d_transpose": getattr(ly, "__conv3d_transpose_layer__"),
            "lstm": getattr(ly, "__lstm_layer__"),
            "dense": getattr(ly, "__linear_layer__")
        }

        self.forward_tensor = self.__layers_block__()
        if isinstance(self.forward_tensor, list):
            self.layers_list = self.forward_tensor
        
        else:
            self.layers_list = [self.forward_tensor]
            
        
        if dropout:
            
            layer = Dropout(rate=dropout_rate)
            self.layers_list.append(layer)
        
        if normalization:
             
            layer = LayerNormalization(epsilon=epsilon)
            self.layers_list.append(layer)
    
    def __get_layers_n__(self, params):

        for param in params.values():
            
            try:
                
                layers_n = len(param)
                return layers_n

            except BaseException:
                pass
        
    def __layers_block__(self):
        
        layer_type = self.layers_params["LayerType"]
        params = self.layers_params["params"]

        weights_params = None
        if "weights_init" in self.layers_params.keys():
            weights_params = self.layers_params["weights_init"]

        layers_n = self.__get_layers_n__(params=params)
        if layers_n != 1:
            
            forward_tensor = []
            for layer_n in range(layers_n):
                
                if self.layers_params["LayerType"] == "lstm":
                    layer = self.__layer_generation__[layer_type](params=params, layer_n=layer_n, norm_params=self.norm_params, layers_n=layers_n)

                else:
                    layer = self.__layer_generation__[layer_type](params=params, weights_params=weights_params, layer_n=layer_n, norm_params=self.norm_params)

                forward_tensor.append(layer)
        
        else:
            forward_tensor = self.__layer_generation__[layer_type](params=params, weights_params=weights_params, norm_params=self.norm_params)
        
        return forward_tensor
                
    def __get_weights_initializer__(self, layer, weights_params):
        
        if "weights_init" in self.layers_params.keys():

            init_type = weights_params["init_type"]
            params = weights_params["params"]

            if init_type == "random_normal":
                weights_init = RandomNormal(mean=params["mean"], stddev=params["stddev"])
            
            elif init_type == "random_uniform":
                weights_init = RandomUniform(minval=params["min_val"], maxval=params["max_val"])
            
            layer.kernel_initializer = weights_init


        
    def __call__(self, x):
         
        for layer in self.layers_list:
            x = layer(x)
        
        return x
    
    

        











    
    
    

    
