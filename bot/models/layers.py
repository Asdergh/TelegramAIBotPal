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



def __norm_and_dropout__(layer, norm_params):
    
    normalized_layer = Sequential(layers=[
        layer,
        Dropout(rate=norm_params[0]),
        LayerNormalization(epsilon=norm_params[1])
    ])

    return normalized_layer

    
def __get_weights_initializer__(layer, weights_params):
        

        init_type = weights_params["init_type"]
        params = weights_params["params"]

        if init_type == "random_normal":
            weights_init = RandomNormal(mean=params["mean"], stddev=params["stddev"])
            
        elif init_type == "random_uniform":
            weights_init = RandomUniform(minval=params["min_val"], maxval=params["max_val"])
            
        layer.kernel_initializer = weights_init


def __linear_layer__(params, weights_params, layer_n=0, norm_params=None):

    linear_layer = Dense(units=params["units"][layer_n], activation=params["activations"][layer_n])
    if weights_params is not None:
        __get_weights_initializer__(layer=linear_layer, weights_params=weights_params)
    
    if norm_params is not None:
       linear_layer = __norm_and_dropout__(norm_params=norm_params)

    return linear_layer
        

def __conv1d_layer__(params, weights_params, layer_n=0, norm_params=None):
        
    conv_layer = Conv1D(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], strides=params["strides"][layer_n], padding=params["padding"][layer_n])
    if weights_params is not None:
        __get_weights_initializer__(layer=conv_layer, weights_params=weights_params) 
    
    conv_layer = Sequential(layers=[conv_layer, Activation(params["activations"][layer_n])])
    if "polling" in params.keys():
        if params["pooling"][layer_n]:

            conv_layer = Sequential(layers=[conv_layer, MaxPool1D(pool_size=params["pooling_size"])])
        
    
        
    if norm_params is not None:
        conv_layer = __norm_and_dropout__(norm_params=norm_params)

    return conv_layer
    

def __conv2d_layer__(params, weights_params, layer_n=0, norm_params=None):
        
    conv_layer = Conv2D(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], strides=params["strides"][layer_n], padding=params["padding"][layer_n])
    if weights_params is not None:
        __get_weights_initializer__(layer=conv_layer, weights_params=weights_params) 


    conv_layer = Sequential(layers=[conv_layer, Activation(params["activations"][layer_n])])
    if "polling" in params.keys():
        if params["pooling"][layer_n]:

            conv_layer = Sequential(layers=[conv_layer, MaxPool2D(pool_size=params["pooling_size"])])

    if norm_params is not None:
        conv_layer = __norm_and_dropout__(norm_params=norm_params)
    
    return conv_layer
    

def __conv3d_layer__(params, weights_params, layer_n=0, norm_params=None):
        
    conv_layer = Conv3D(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], strides=params["strides"][layer_n], padding=params["padding"][layer_n])
    if weights_params is not None:
        __get_weights_initializer__(layer=conv_layer, weights_params=weights_params) 

    
        conv_layer = Sequential(layers=[conv_layer, Activation(params["activations"][layer_n])])
    if "polling" in params.keys():
        if params["pooling"][layer_n]:

            conv_layer = Sequential(layers=[conv_layer, MaxPool3D(pool_size=params["pooling_size"])])

    if norm_params is not None:
        conv_layer = __norm_and_dropout__(norm_params=norm_params)

    return conv_layer


def __conv1d_transpose_layer__(params, weights_params, layer_n=0, norm_params=None):
        
    conv_layer = Conv1DTranspose(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], strides=params["strides"][layer_n], padding=params["padding"][layer_n])
    if weights_params is not None:
        __get_weights_initializer__(layer=conv_layer, weights_params=weights_params) 

    
    conv_layer = Sequential(layers=[conv_layer, Activation(params["activations"][layer_n])])
    if "polling" in params.keys():
        if params["pooling"][layer_n]:

            conv_layer = Sequential(layers=[conv_layer, MaxPool1D(pool_size=params["pooling_size"])])
        
    
    if norm_params is not None:
        conv_layer = __norm_and_dropout__(norm_params=norm_params)

    return conv_layer
    

def __conv2d_transpose_layer__(params, weights_params, layer_n=0, norm_params=None):
        
    conv_layer = Conv2DTranspose(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], strides=params["strides"][layer_n], padding=params["padding"][layer_n])
    if weights_params is not None:
        __get_weights_initializer__(layer=conv_layer, weights_params=weights_params) 

    conv_layer = Sequential(layers=[conv_layer, Activation(params["activations"][layer_n])])
    if "polling" in params.keys():
        if params["pooling"][layer_n]:

            conv_layer = Sequential(layers=[conv_layer, MaxPool2D(pool_size=params["pooling_size"])])

    if norm_params is not None:
        conv_layer = __norm_and_dropout__(norm_params=norm_params)

    return conv_layer
    

def __conv3d_transpose_layer__(params, weights_params, layer_n=0, norm_params=None):
        
    conv_layer = Conv3DTranspose(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], strides=params["strides"][layer_n], padding=params["padding"][layer_n])
    if weights_params is not None:
        __get_weights_initializer__(layer=conv_layer, weights_params=weights_params) 

        conv_layer = Sequential(layers=[conv_layer, Activation(params["activations"][layer_n])])
    if "polling" in params.keys():
        if params["pooling"][layer_n]:

            conv_layer = Sequential(layers=[conv_layer, MaxPool3D(pool_size=params["pooling_size"])])

    if norm_params is not None:
        conv_layer = __norm_and_dropout__(norm_params=norm_params)

    return conv_layer


def __lstm_layer__(params, layer_n=0, norm_params=None, layers_n=1):

    
    if layers_n != 1:

        if layer_n == layers_n - 1:
            rq = params["return_sequences"]
        
        else:
            rq = True
    
    else:
        rq = params["return_sequences"]


    lstm_layer = LSTM(units=params["units"][layer_n], return_sequences=rq)    
    if params["bi"][layer_n]:
            lstm_layer = Bidirectional(lstm_layer)
                
    lstm_layer = Sequential(layers=[lstm_layer, Activation(params["activations"][layer_n])])
    if norm_params is not None:
        lstm_layer = __norm_and_dropout__(norm_params=norm_params)
    
    

    return lstm_layer