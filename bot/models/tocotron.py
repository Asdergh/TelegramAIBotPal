import numpy as np
import random as rd
import matplotlib.pyplot as plt
import json as js
import tensorflow.keras.backend as K
import os

from tensorflow.keras.layers import Input, Dropout, Concatenate, Dense, RepeatVector
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Conv1D, Conv2D
from tensorflow.keras.layers import LayerNormalization, Attention, Activation
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import Model

class Tocotron:

    def __init__(self, params_json=None, params_path=None) -> None:
        
        self.params_json = params_json
        if params_path is None:
                params_path = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\Tacotron2.json"

        if self.params_json is None:
            self._load_params_(filepath=params_path)
        
        self._save_params_(filepath=params_path)
        self._build_encoder_()
        self._build_decoder_()
        self._build_model_()

    

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
    

    def _build_decoder_(self):

        decoder_params = self.params_json["decoder_params"]

        input_layer = Input(shape=(None, decoder_params["filters"][-1]))
        encoder_output_layer = Input(shape=(None, decoder_params["filters"][-1], ))
        
        prenet_layer = Sequential(layers=[
                                    LayerNormalization(epsilon=decoder_params["prenet_params"]["epsilon"]),
                                    Dropout(rate=decoder_params["prenet_params"]["dropout_rate"])])(input_layer)

        input_conv_layer = Conv1D(filters=decoder_params["filters"][0], kernel_size=decoder_params["kernel_size"][0],
                                padding=decoder_params["padding"][0], strides=decoder_params["strides"][0])(prenet_layer)
        
        lstm_layer = LSTM(units=self.params_json["encoder_params"]["units"], return_sequences=True)(prenet_layer)
        att_layer = Attention()([lstm_layer, encoder_output_layer])
        lstm_layer = LSTM(units=self.params_json["encoder_params"]["units"])(att_layer)

        linear_projection_layer = Sequential(layers=[
                    Dense(units=self.params_json["encoder_params"]["units"], activation=decoder_params["linear_projection"]["activation"]),
                    Dropout(rate=decoder_params["linear_projection"]["dropout_rate"]),
                    LayerNormalization(epsilon=decoder_params["linear_projection"]["epsilon"]),
                    RepeatVector(n=decoder_params["mel_record_len"])])(lstm_layer)
        
        conv_layer = Concatenate()([input_conv_layer, linear_projection_layer])
        
        
        for layer_n in range(len(decoder_params["filters"])):

            conv_layer = Conv2D(filters=1, kernel_size=decoder_params["kernel_size"][layer_n],
                                padding=decoder_params["padding"][layer_n], strides=decoder_params["strides"][layer_n])(conv_layer)
            
            conv_layer = Sequential(layers=[Activation(decoder_params["activations"][layer_n]),
                                            LayerNormalization(epsilon=decoder_params["normalization_epsilon"]),
                                            Dropout(rate=decoder_params["dropout_rates"][layer_n])])(conv_layer)
            
        
        self.decoder = Model(inputs=[input_layer, encoder_output_layer], outputs=conv_layer)
    
    def _build_model_(self):

        encoder_input_layer = Input(shape=(None, ))
        decoder_input_layer = Input(shape=(None, self.params_json["decoder_params"]["filters"][-1], ))
        
        encoder_output = self.encoder(encoder_input_layer)
        decoder_output = self.decoder([decoder_input_layer, encoder_output])
        
        self.model = Model(inputs=[encoder_input_layer, decoder_input_layer], outputs=decoder_output)
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
    
    def train(self, encoder_train_samples, decoder_train_samples, epochs, batch_size, shuffle):


        weights_folder = os.path.join(self.params_json["run_folder"], "weights.weights.h5")
        self.model.fit([encoder_train_samples, decoder_train_samples], 
                       decoder_train_samples, epochs=epochs, 
                       batch_size=batch_size, shuffle=shuffle)
        self.model.save_weights(filepath=weights_folder)
    
    def text_to_mel(self, input_text):

        input_tokens = self.encoder_tokenizer.texts_to_sequences([input_text])
        input_tokens = np.expand_dims(np.asarray(input_tokens), axis=0)
        encoder_output = self.encoder.predict(input_tokens)
        
        if self.params_json["random_init"]:
            decoder_input = np.random.normal(0, 1.98, (1,
                                                       self.params_json["decoder_params"]["mel_sequence_len"], 
                                                       self.params_json["decoder_params"]["filters"][-1]))
        
        if self.params_json["random_init"]:
            decoder_input = np.zeros((1, 
                                    self.params_json["decoder_params"]["mel_sequence_len"], 
                                    self.params_json["decoder_params"]["filters"][-1]))
        
        decoded_output = self.decoder.predict([decoder_input, encoder_output])
        return decoded_output
    
    

        


    
    

        
    

        
        
        
    
        