import numpy as np
import matplotlib.pyplot as plt
import json as js
import os
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, BatchNormalization
from tensorflow.keras.layers import Embedding, LSTM, GRU, Bidirectional, RepeatVector
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model




class RnnConv:

    def __init__(self, params_json=None, params_path=None) -> None:
        
        self.params_json = params_json
        if self.params_json is None:
            self._load_params_(filepath=params_path)

        self.conv_layers_n = len(self.params_json["conv_params"]["filters"])
        self.rnn_layers_n = len(self.params_json["rnn_params"]["units"])

        self._build_conv_model_()
        self._build_rnn_model_()
        self._build_model_()
    
    def _load_params_(self, filepath):
        
        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\RnnConv.json"

        with open(filepath, "r") as json_file:
            self.params_json = js.load(json_file)
    
    def _save_params(self, filepath):

        with open(filepath, "w") as json_file:
            js.dump(json_file, self.params_json)

    def _build_conv_model_(self):
        
        conv_params = self.params_json["conv_params"]
        input_layer_conv = Input(shape=self.params_json["input_shape"])
        conv_layer = input_layer_conv

        for layer_n in range(self.conv_layers_n):

            conv_layer = Conv2D(filters=conv_params["filters"][layer_n],
                                kernel_size=conv_params["kernel_size"][layer_n],
                                strides=conv_params["strides"][layer_n],
                                padding="same")(conv_layer)
            
            conv_layer = Activation(conv_params["activations"][layer_n])(conv_layer)
            if not conv_params["single_dropout"]:
                conv_layer = Dropout(conv_params["dropout_rates"][layer_n])(conv_layer)
            
            conv_layer = BatchNormalization()(conv_layer)
        
        dense_layer = Flatten()(conv_layer)        
        if conv_params["dense_units"] != 0:

            for layer_n in range(len(conv_params["dense_units"])):

                dense_layer = Dense(units=conv_params["dense_units"][layer_n], activation=conv_params["dense_activations"][layer_n])(dense_layer)
                dense_layer = Dropout(rate=conv_params["dense_dropout_rates"][layer_n])(dense_layer)
            
        
        self.state_h_layer = Dense(units=self.params_json["rnn_params"]["units"][0], activation=conv_params["output_activation"])(dense_layer)
        self.state_c_layer = Dense(units=self.params_json["rnn_params"]["units"][0], activation=conv_params["output_activation"])(dense_layer)
        self.conv_model = Model(inputs=input_layer_conv, outputs=[self.state_h_layer, self.state_c_layer])

    def _build_rnn_model_(self):

        rnn_params = self.params_json["rnn_params"]

        input_layer_rnn = Input(shape=(None, ))
        input_state_h_layer = Input(shape=(rnn_params["units"][0], ))
        input_state_c_layer = Input(shape=(rnn_params["units"][0], ))
        embedding_layer = Embedding(input_dim=self.params_json["total_labels_n"], output_dim=rnn_params["embedding_dim"])(input_layer_rnn)
        
        for layer_n in range(self.rnn_layers_n):
            
            if layer_n == 0:     
                lstm_layer = LSTM(units=rnn_params["units"][layer_n], return_sequences=True)(embedding_layer, initial_state=[input_state_h_layer, input_state_c_layer])
            

            elif layer_n != (self.rnn_layers_n - 1):
                lstm_layer = LSTM(units=rnn_params[layer_n], return_sequences=True)(lstm_layer)
            
            if not rnn_params["single_dropout"]:
                lstm_layer = Dropout(rate=rnn_params["dropout_rates"][layer_n])(lstm_layer)
        
        lstm_layer = LSTM(units=rnn_params["units"][-1])(lstm_layer)
        if rnn_params["single_dropout"]:
            lstm_layer = Dropout(rate=rnn_params["dropout_rates"][layer_n])(lstm_layer)
        
        self.output_layer = Dense(units=self.params_json["total_labels_n"], activation="softmax")(lstm_layer)
        self.rnn_model = Model(inputs=[input_layer_rnn, input_state_h_layer, input_state_c_layer], outputs=self.output_layer)

    def _build_model_(self):
        
        sequence_input = Input(shape=(None, ))
        model_input_layer = Input(shape=self.params_json["input_shape"])
        
        state_h_output, state_c_output = self.conv_model(model_input_layer)
        model_output_layer = self.rnn_model([sequence_input, state_h_output, state_c_output])
        
        self.model = Model(inputs=[model_input_layer, sequence_input], outputs=model_output_layer)
        self.model.compile(loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="rmsprop")

        



if __name__ == "__main__":


    input_image_tensor = np.random.normal(0.0, 1.0, (100, 224, 224, 3))
    random_values = np.random.randint(0, 1000, 100)
    input_rnn_tensor = np.asarray([np.random.choice(a=random_values, size=30) for _ in range(100)], dtype="int64")
    random_categorical = to_categorical(random_values)
    print(random_categorical.shape)

    params_json = {
        "total_labels_n": random_categorical.shape[1],
        "input_shape": (224, 224, 3),
        "conv_params": {

            "single_dropout": False,
            "filters": [128, 64, 64, 32],
            "strides": [1, 2, 2, 1],
            "kernel_size": [3, 3, 3, 3],
            "activations": ["linear", "linear", "linear", "linear"],
            "dropout_rates": [0.26, 0.26, 0.26, 0.26],

            "dense_units": [32, 64, 64, 128],
            "dense_dropout_rates": [0.26, 0.26, 0.26, 0.26],
            "dense_activations": ["linear", "linear", "relu", "relu"],
            
            "output_activation": "tanh",
        },

        "rnn_params": {
            
            "embedding_dim": 100,
            "single_dropout": False,
            "units": [256, 256],
            "dropout_rates": [0.26, 0.26]
        }
    }


    img2seq_model = RnnConv(params_json=params_json)
    

    img2seq_model.model.fit([input_image_tensor, input_rnn_tensor], random_categorical, epochs=1, batch_size=32)
    encoder_output = img2seq_model.conv_model.predict(input_image_tensor)
    categorical_output = img2seq_model.rnn_model.predict([input_rnn_tensor, encoder_output[0], encoder_output[1]])

    print(categorical_output)
    

    




    


