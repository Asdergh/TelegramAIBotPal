import numpy as np
import matplotlib.pyplot as plt
import random as rd
import json as js
import os
import tensorflow as tf
import tensorflow.nn as tn
import cv2

from tensorflow.nn import moments
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, UpSampling2D, Conv2D, Dropout, Input, Normalization
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


class STF(Model):


    def __init__(self, input_shape=(128, 128, 3), **kwargs):

        super().__init__(**kwargs)
        self.input_shape = input_shape

        self.encoder = self.__encoder__()
        self.decoder = self.__decoder__()
        self.loss_net = self.__loss_net__()

        print(self.encoder.summary())
        print(self.decoder.summary())
        print(self.loss_net.summary())
    

    def __get_moments__(self, sample):

        mean, variance = tn.moments(sample, axes=[1, 2], keepdims=True)
        standard_division = tf.sqrt(variance)
        
        return mean, standard_division
    

    def __adain__(self, content, style):
        
        
        content_mean, content_std = self.__get_moments__(content)
        style_mean, style_std = self.__get_moments__(style)
        t = style_std * ((content - content_mean) / content_std) + style_mean

        return t
    

    def __loss_net__(self):
        
        vgg19 = VGG19(include_top=False, weights="imagenet", input_shape=self.input_shape)
        vgg19.trainable = False

        need_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
        outputs = [vgg19.get_layer(layer_name).output for layer_name in need_layers]
        mini_vgg_net = Model(inputs=vgg19.input, outputs=outputs)

        input_layer = Input(shape=self.input_shape)
        output_layer = mini_vgg_net(input_layer)

        loss_net = Model(inputs=input_layer, outputs=output_layer)
        return loss_net
    
    def __encoder__(self):

        vgg19 = VGG19(include_top=False, weights="imagenet", input_shape=self.input_shape)
        vgg19.trainable = False
        mini_vgg_net = Model(inputs=vgg19.inputs, outputs=vgg19.get_layer("block4_conv1").output)
        
        input_layer = Input(shape=self.input_shape)
        output_layer = mini_vgg_net(input_layer)

        encoder = Model(inputs=input_layer, outputs=output_layer, name="encoder_net")
        return encoder

    def __decoder__(self):

        configs = [
            ["conv", 512, {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"} ],
            ["up_sampling", 2, {"data_format":None, "interpolation": "nearest"}                     ],
            ["conv", 256, {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"} ],
            ["conv", 256, {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"} ],
            ["conv", 256, {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"} ],
            ["conv", 256, {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"} ],
            ["up_sampling", 2, {"data_format":None, "interpolation": "nearest"}                     ],
            ["conv", 128, {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"} ],
            ["conv", 128, {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"} ],
            ["up_sampling", 2, {"data_format":None, "interpolation": "nearest"}                     ],
            ["conv", 64, {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}  ],
            ["conv", 3, {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}   ],
        ]
        layers = {
            "conv": Conv2D,
            "up_sampling": UpSampling2D
        }
        
        input_layer = Input((None, None, 512))
        layer = input_layer
        for conf in configs:
            layer = layers[conf[0]](conf[1], **conf[2])(layer)
        
        decoder = Model(inputs=input_layer, outputs=layer)
        return decoder


    def compile(self, optimizer, loss_fn):

        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.style_loss_tracker = Mean(name="style_loss")
        self.content_loss_tracker = Mean(name="content_loss")
        self.total_loss_tracker = Mean(name="total_loss")
    
    def train_step(self, inputs):
        

        content, style = inputs
        content_loss = 0.0
        style_loss = 0.0
        total_loss = 0.0

        with tf.GradientTape() as tape:

            content_encoding = self.encoder(content)
            style_encoding = self.encoder(style)

            t = self.__adain__(content_encoding, style_encoding)
            reconstracted_image = self.decoder(t)
            
            rec_vgg_features = self.loss_net(reconstracted_image)
            style_vgg_features = self.loss_net(style)

            content_loss = self.loss_fn(t, rec_vgg_features[-1])
            
            for (input, output) in zip(style_vgg_features, rec_vgg_features):

                input_mean, input_std = self.__get_moments__(input)
                output_mean, output_std = self.__get_moments__(output)
                style_loss += self.loss_fn(input_mean, output_mean) + self.loss_fn(input_std, output_std)
            
            total_loss = content_loss + style_loss

        trainable_weights = self.decoder.trainable_variables
        gradient = tape.gradient(total_loss, trainable_weights)
        self.optimizer.apply_gradients(zip(gradient, trainable_weights))

        self.style_loss_tracker.update_state(style_loss)
        self.content_loss_tracker.update_state(content_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result()
        }

    @property
    def metrics(self):

        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker
        ]
                

            


  

