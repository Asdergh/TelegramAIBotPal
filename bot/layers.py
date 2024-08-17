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
        
        random_idx = np.random.randint(0, input_layer.shape[0], self.samples_n)
        samples = input_layer[random_idx]
        sequence = samples[0]
        
        for  sample_number in range(1, self.samples_n):
            
            sample = samples[sample_number]
            random_coeff = np.random.normal(0, 1, sequence.shape[0])
            sequence += sample * random_coeff
        
        lstm_layer = LSTM(units=self.units, return_sequences=True)(sequence)
        return self.lstm_layer

    def compute_output_shape(self):
        return (None, None, self.units) 


