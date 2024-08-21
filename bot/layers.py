import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout, Layer, Conv1D, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, BatchNormalization, Multiply
from tensorflow.keras.layers import Embedding, LSTM, GRU, Bidirectional, RepeatVector, Concatenate
from tensorflow.keras import Model

class ConvSequence(Layer):

    def __init__(self, units, samples_n, sequence_l):

        super(ConvSequence, self).__init__()
        self.units = units
        self.samples_n = samples_n
        self.sequence_l = sequence_l
    
    def call(self, input_layer):
        
        layer = Flatten()(input_layer)
        layer = Dense(units=self.units, activation="softmax")(layer)    
        return layer
    
    def compute_output_shape(self):
        return (None, None, self.units)

class WaveNetLayer(Layer):

    def __init__(self, filters, kernel_size, output_dim, max_seq_len=1):

        super(WaveNetLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.max_sequence_lenght = max_seq_len
        self.padding = "causal"

        self.conv_dialation_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.tanh_layer = Activation("tanh")
        self.sigma_layer = Activation("sigmoid")
        self.gate_layer = Multiply()
        self.flatten_layer = Flatten()
        self.dense_layer = Dense(units=self.output_dim, activation="softmax")
        if self.max_sequence_lenght != 1:
            self.repeat_layer = RepeatVector(n=self.max_sequence_lenght)
    
    def call(self, input_layer):
        
        layer = self.conv_dialation_layer(input_layer)
        gate_tanh_layer = self.tanh_layer(layer)
        gate_sigma_layer = self.sigma_layer(layer)
        
        gate_layer = self.gate_layer([gate_tanh_layer, gate_sigma_layer])
        flatten_layer = self.flatten_layer(gate_layer)

        dense_layer = self.dense_layer(flatten_layer)
        if self.max_sequence_lenght != 1:
            dense_layer =  self.repeat_layer(dense_layer)

        return dense_layer
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


