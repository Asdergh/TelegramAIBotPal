import numpy as np
import random as rd
import matplotlib.pyplot as plt
import json as js
import tensorflow.keras.backend as K
import os

from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Conv1D
from tensorflow.keras.layers import LayerNormalization, Attention 
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import AbsoluteError
from tensorflow.keras import Model

class Tocotron:

    def __init__(self, params_json=None, params_path=None) -> None:
        
        self.params_json = params_json
        if self.params_json is None:
            
            if params_path is None:
                params_path = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\Tacotron2.json"
            self._load_params_(filepath=params_json)
        
        self._save_params_(filepath=params_json)
    

    def _load_tokenizer_(self, tokenizer):

        self.encoder_tokenizer = tokenizer
        encoder_vocab_path = os.path.join(self.params_json["run_folder"], "encoder_vocab.json")

        with open(encoder_vocab_path, "w") as json_file:
            js.dump(self.encoder_tokenizer, json_file)
    
    def _expand_tokenizer_(self, new_words):
        
        encoder_vocab_path = os.path.join(self.params_json["run_folder"], "encoder_vocab.json")
        with open(encoder_vocab_path, "r") as json_file:
            word_index = js.load(json_file)

        for word in new_words:

            if word not in word_index.keys():
                word_index[word] = word_index.values()[-1] + 1
        
        self.tokenizer.word_index = word_index
        
    def _load_params_(self, filepath=None):

        with open(filepath, "r") as json_file:
            self.params_json = js.load(json_file)
    
    def _save_params_(self, filepath):

        with open(filepath, "w") as json_file:
            js.dump(self.params_json, json_file)
    
    def _build_encoder_(self):

        encoder_params = self.params_json["encoder_params"]

        input_layer = Input(shape=(None, ))
        embedding_layer = Embedding(input_dim=encoder_params["total_words_n"], output_dim=encoder_params["embedding_size"])(input_layer)
        lstm_layer = embedding_layer

        for _ in range(encoder_params["layers_n"]):

            lstm_layer = LSTM(units=encoder_params["units"], return_sequences=True)(lstm_layer)
            lstm_layer = Dropout(rate=encoder_params["dropout_rate"])(lstm_layer)
        
        self.encoder = Model(inputs=input_layer, outputs=lstm_layer)
    

    def build_decoder(self):

        decoder_params = self.params_json["decoder_params"]

        input_layer = Input(shape=(None, decoder_params["filters"][-1]))
        encoder_output_layer = Input(shape=(None, decoder_params["filters"][-1], ))
        
        prenet_layer = Sequential(layers=[
            LayerNormalization(epsilon=decoder_params["prenet_params"]["epsilon"]),
            Dropout(rate=decoder_params["prenet_params"]["dropout_rate"])])(input_layer)

        lstm_layer = LSTM(units=self.params_json["encoder_params"]["units"], return_sequences=True)(prenet_layer)
        att_layer = Attention()([lstm_layer, encoder_output_layer])

        conv_layer = att_layer
        for layer_n in range(len(decoder_params["filters"])):

            conv_layer = Conv1D(filters=decoder_params["filters"][layer_n], kernel_size=decoder_params["kernel_size"][layer_n],
                                padding=decoder_params["padding"][layer_n], strides=decoder_params["strides"][layer_n])(conv_layer)
            
            conv_layer = LayerNormalization(epsilon=decoder_params["normalization_epsilon"])(conv_layer)
            conv_layer = Dropout(rate=decoder_params["dropout_rates"][layer_n])(conv_layer)
        
        self.decoder = Model(inputs=[input_layer, encoder_output_layer], outputs=conv_layer)
    
    def _build_model_(self):

        encoder_input_layer = Input(shape=(None, ))
        decoder_input_layer = Input(shape=(None, self.params_json["decoder_params"]["filters"][-1], ))
        
        encoder_output = self.encoder(encoder_input_layer)
        decoder_output = self.decoder([decoder_input_layer, encoder_output])
        
        self.model = Model(inputs=[encoder_input_layer, decoder_input_layer], outputs=decoder_output)
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
    
    def _train_function_(self, encoder_train_samples, decoder_train_samples, epochs, batch_size, shuffle):


        weights_folder = os.path.join(self.params_json["run_folder"], "weights.weights.h5")
        self.model.fit([encoder_train_samples, decoder_train_samples], 
                       decoder_train_samples, epochs, 
                       batch_size, shuffle)
        self.model.save_weights(filepath=weights_folder)
    
    def text_to_mel(self, input_text):

        


    
    

        
    

        
        
        
    
        