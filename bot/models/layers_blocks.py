import numpy as np
import random as rd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Flatten, Dense
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dropout, LayerNormalization, Reshape
from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Zeros
from tensorflow.keras import Model, Sequential
from tensorflow import Module


linear_params = {
    "LayerType": "dense",
    "params":{
        "units": [32, 32, 64, 128],
        "activations": ["tanh", "tanh", "linear", "relu"]
    }

}

class LayersBlock(Module):

    def __init__(self, layers_params, name=None, 
                normalization=True, dropout=True,
                dropout_rate=0.56, epsilon=0.01):
        
        super().__init__(name)
        self.layers_params = layers_params
        self.__block_generation__ = {
            "conv1d": self._conv1d_block_,
            "conv2d": self._conv2d_block_,
            "conv3d": self._conv3d_block_,
            "conv1d_transpose": self._conv1d_transpose_block_,
            "conv2d_transpose": self._conv2d_transpose_block_,
            "conv3d_transpose": self._conv3d_transpose_block_,
            "lstm": self._lstm_block_,
            "dense": self._linear_block_
        }

        layer_type = self.layers_params["LayerType"]
        params = self.layers_params["params"]
        self.forward_tensor = self.__block_generation__[layer_type](params=params)
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
    

    def __check_params__(self, params):

        params_list = [params[param] for param in params]
        random_param = rd.choice(params_list)
        for param in params_list:

            if param is list:
                
                if type(param) == type(random_param):
                    pass

                else:
                    raise ValueError("all params must have [list or none list] types !!!")
                
    def __get_weights_initializer__(self, params, layer):
        
        if "weights_init" in params.keys():

            init_type = params["weights_init"]["init_type"]
            params = params["weights_init"]["params"]

            if init_type == "random_normal":
                weights_init = RandomNormal(mean=params["mean"], stddev=params["stddev"])
            
            elif init_type == "random_uniform":
                weights_init = RandomUniform(minval=params["min_val"], maxval=params["max_val"])
            
            layer.kernel_initializer = weights_init


        
    def __call__(self, x):
         
        for layer in self.layers_list:
            x = layer(x)
        
        return x
    

    def __conv1d_layer__(self, params, layer_n=0):
        
        conv_layer = Conv1D(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], strides=params["strides"][layer_n], padding=params["padding"][layer_n])
        self.__get_weights_initializer__(params=params, layer=conv_layer) 
               
        conv_layer = Sequential(layers=[conv_layer, Activation(params["activation"][layer_n])])
        return conv_layer
        

    def _linear_block_(self, params):
     
        layers_n = len(params["units"])
        if layers_n != 1:
            
            forward_tensor = []
            for layer_n in range(layers_n):
                
                linear_layer = Dense(units=params["units"][layer_n], activation=params["activations"][layer_n])
                if "weights_init" in params.keys():
                    
                    weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                    linear_layer.kernel_initializer = weights_init

                forward_tensor.append(linear_layer)
        
        else:

            linear_layer = Dense(units=params["units"], activation=params["activations"])
            if "weights_init" in params.keys():
                    
                weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                linear_layer.kernel_initializer = weights_init

            forward_tensor = linear_layer
               
        return forward_tensor
    
    def _conv1d_block_(self, params):
            
            layers_n = len(params["filters"])
            if layers_n:

                forward_tensor = []
                for layer_n in range(layers_n):
                
                    
                    conv_layer = Conv1D(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], 
                                        strides=params["strides"][layer_n], padding=params["padding"][layer_n])
                    
                    if "weights_init" in params.keys():
                
                        weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                        conv_layer.kernel_initializer = weights_init  
                    
                    conv_layer = Sequential(layers=[conv_layer, Activation(params["activation"][layer_n])])
                    forward_tensor.append(conv_layer)
            
            else:
                    
                forward_tensor = Conv1D(filters=params["filters"][0], kernel_size=params["kernel_size"][0], strides=params["strides"][0], padding=params["padding"][0])
                if "weights_init" in params.keys():
                
                    weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                    forward_tensor.kernel_initializer = weights_init  
                
                forward_tensor = Sequential(layers=[forward_tensor, Activation(params["activation"][0])])

            return forward_tensor



    def _conv2d_block_(self, params):
            
            layers_n = len(params["filters"])            
            if layers_n:

                forward_tensor = []
                for layer_n in range(layers_n):
                
                    
                    conv_layer = Conv2D(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], 
                                        strides=params["strides"][layer_n], padding=params["padding"][layer_n])
                    
                    if "weights_init" in params.keys():
                
                        weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                        conv_layer.kernel_initializer = weights_init  
                    
                    conv_layer = Sequential(layers=[conv_layer, Activation(params["activation"][layer_n])])
                    forward_tensor.append(conv_layer)
            
            else:
                    
                forward_tensor = Conv2D(filters=params["filters"][0], kernel_size=params["kernel_size"][0], strides=params["strides"][0], padding=params["padding"][0])
                if "weights_init" in params.keys():
                
                    weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                    forward_tensor.kernel_initializer = weights_init  
                
                forward_tensor = Sequential(layers=[forward_tensor, Activation(params["activation"][0])])
            return forward_tensor


    def _conv3d_block_(self, params):
            
            layers_n = len(params["filters"])        
            if layers_n:

                forward_tensor = []
                for layer_n in range(layers_n):
                
                    
                    conv_layer = Conv3D(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], 
                                        strides=params["strides"][layer_n], padding=params["padding"][layer_n])
                    
                    if "weights_init" in params.keys():
                
                        weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                        conv_layer.kernel_initializer = weights_init  
                    
                    conv_layer = Sequential(layers=[conv_layer, Activation(params["activation"][layer_n])])
                    forward_tensor.append(conv_layer)
            
            else:
                    
                forward_tensor = Conv3D(filters=params["filters"][0], kernel_size=params["kernel_size"][0], strides=params["strides"][0], padding=params["padding"][0])
                if "weights_init" in params.keys():
                
                    weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                    forward_tensor.kernel_initializer = weights_init  
                
                forward_tensor = Sequential(layers=[forward_tensor, Activation(params["activation"][0])])
            return forward_tensor

    def _conv1d_transpose_block_(self, params):
        
        layers_n = len(params["filters"])
        if layers_n:

                forward_tensor = []
                for layer_n in range(layers_n):
                
                    
                    conv_layer = Conv1DTranspose(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], 
                                        strides=params["strides"][layer_n], padding=params["padding"][layer_n])
                    
                    if "weights_init" in params.keys():
                
                        weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                        conv_layer.kernel_initializer = weights_init  
                    
                    conv_layer = Sequential(layers=[conv_layer, Activation(params["activation"][layer_n])])
                    forward_tensor.append(conv_layer)
            
        else:
                    
            forward_tensor = Conv1DTranspose(filters=params["filters"][0], kernel_size=params["kernel_size"][0], strides=params["strides"][0], padding=params["padding"][0])
            if "weights_init" in params.keys():
                
                weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                forward_tensor.kernel_initializer = weights_init  
            
            forward_tensor = Sequential(layers=[forward_tensor, Activation(params["activation"][0])])
        return forward_tensor


    def _conv2d_transpose_block_(self, params):
        
        layers_n = len(params["filters"])
        if layers_n != 1:

                forward_tensor = []
                for layer_n in range(layers_n):
                
                    
                    conv_layer = Conv2DTranspose(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], 
                                        strides=params["strides"][layer_n], padding=params["padding"][layer_n])
                    
                    if "weights_init" in params.keys():
                
                        weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                        conv_layer.kernel_initializer = weights_init  
                    
                    conv_layer = Sequential(layers=[conv_layer, Activation(params["activation"][layer_n])])
                    forward_tensor.append(conv_layer)
            
        else:
                    
            forward_tensor = Conv2DTranspose(filters=params["filters"][0], kernel_size=params["kernel_size"][0], strides=params["strides"][0], padding=params["padding"][0])
            if "weights_init" in params.keys():
                
                weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                forward_tensor.kernel_initializer = weights_init 
            
            forward_tensor = Sequential(layers=[forward_tensor, Activation(params["activation"][0])])
                
        return forward_tensor


    def _conv3d_transpose_block_(self, params):
        
        layers_n = len(params["filters"])    
        if layers_n != 1:

                forward_tensor = []
                for layer_n in range(layers_n):
                
                    
                    conv_layer = Conv3DTranspose(filters=params["filters"][layer_n], kernel_size=params["kernel_size"][layer_n], strides=params["strides"][layer_n], padding=params["padding"][layer_n])
                    if "weights_init" in params.keys():
                
                        weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                        conv_layer.kernel_initializer = weights_init  
                    
                    conv_layer = Sequential(layers=[conv_layer, Activation(params["activation"][layer_n])])
                    forward_tensor.append(conv_layer)
            
        else:
                    
            forward_tensor = Conv3DTranspose(filters=params["filters"][0], kernel_size=params["kernel_size"][0], strides=params["strides"][0], padding=params["padding"][0])
            if "weights_init" in params.keys():
                
                weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                forward_tensor.kernel_initializer = weights_init 
            
            forward_tensor = Sequential(layers=[forward_tensor, Activation(params["activation"][0])])
                
        return forward_tensor


    def _lstm_block_(self, params):

        
        layers_n = len(params["units"])
        if layers_n != 1:
            
            forward_tensor = []
            
            for layer_n in range(layers_n - 1):
                
                lstm_layer = LSTM(units=params["units"][layer_n], return_sequences=True)
                if params["bi"]:
                    lstm_layer = Bidirectional(lstm_layer)
                
                if "weights_init" in params.keys():
                    
                    weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                    lstm_layer.kernel_initializer = weights_init

                layer = Sequential(layers=[lstm_layer, Activation(params["activations"][layer_n])])
                forward_tensor.append(layer)
        
            if not params["return_sequences"]:

                output_layer = LSTM(units=params["units"][-1])
                forward_tensor.append(output_layer)
        
        else:

            rq = params["return_sequences"]
            lstm_layer = LSTM(units=params["units"][0], return_sequences=rq)

            if params["bi"]:
                lstm_layer = Bidirectional(lstm_layer)
                
            if "weights_init" in params.keys():
                    
                weights_init = self.__get_weights_initializer__(initializer_params=params["weights_init"])
                lstm_layer.kernel_initializer = weights_init

            forward_tensor = Sequential(layers=[lstm_layer, Activation(params["activations"][0])])
        
        
        return forward_tensor
    
        











    
    
    

    
