import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout, Layer
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, BatchNormalization
from tensorflow.keras.layers import Embedding, LSTM, GRU, Bidirectional, RepeatVector
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


