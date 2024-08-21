import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import json as js
import tensorflow.keras.backend as K


from layers import *
from tensorflow.keras.layers import Input, Dense, Activation, Reshape, LayerNormalization, BatchNormalization, Add
from tensorflow.keras.layers import LSTM, GRU, Masking, Bidirectional, Dropout, Conv2D, Conv2DTranspose, Flatten
from tensorflow.keras.layers import Concatenate, Lambda, Embedding, Multiply, Layer, Conv1D, Attention, RepeatVector
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.random import categorical
from tensorflow import map_fn, expand_dims, convert_to_tensor
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import RandomNormal


class RnnConv:

    def __init__(self, params_json=None, params_path=None) -> None:
        
        self.params_json = params_json
        if self.params_json is None:
            self._load_params_(filepath=params_path)
        
        self.tokenizer = Tokenizer()
        self._save_params_(filepath=params_path)
        self.encoder_layers_n = len(self.params_json["encoder_params"]["filters"])
        self.decoder_layers_n = len(self.params_json["decoder_params"]["units"])
        self.weights_init = RandomNormal(mean=self.params_json["weights_init"]["mean"], stddev=self.params_json["weights_init"]["stddev"])

        self._build_encoder_()
        self._build_decoder_()
        self._build_model_()
        
    
    def _load_params_(self, filepath):
        
        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\RnnConv.json"

        with open(filepath, "r") as json_file:
            self.params_json = js.load(json_file)
    
    def _save_params_(self, filepath):

        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\RnnConv.json"

        with open(filepath, "w") as json_file:
            js.dump(self.params_json, json_file)

    def _build_encoder_(self):
        
        encoder_params = self.params_json["encoder_params"]
        input_layer = Input(shape=self.params_json["input_shape"])
        conv_layer = input_layer

        for layer_n in range(self.encoder_layers_n):

            conv_layer = Conv2D(filters=encoder_params["filters"][layer_n], kernel_size=encoder_params["kernel_size"][layer_n],
                                strides=encoder_params["strides"][layer_n], kernel_initializer=self.weights_init,
                                padding="same")(conv_layer)
            
            conv_layer = Activation(encoder_params["activations"][layer_n])(conv_layer)
            if not encoder_params["single_dropout"]:
                conv_layer = Dropout(encoder_params["dropout_rates"][layer_n])(conv_layer)
            
            conv_layer = BatchNormalization()(conv_layer)
        
        conv_layer = Conv2D(filters=1, kernel_size=3, strides=1)(conv_layer)
        conv_layer = Activation(encoder_params["output_activation"])(conv_layer)

        output_layer = Flatten()(conv_layer)
        output_layer = Dense(units=self.params_json["decoder_params"]["units"][0], activation="tanh")(output_layer)

        self.saved_shape = output_layer.shape[1:]
        self.encoder = Model(inputs=input_layer, outputs=output_layer)

    def _build_decoder_(self):
        
        decoder_params = self.params_json["decoder_params"]
        input_layer = Input(shape=(decoder_params["units"][0], ))
        sequence_input_layer = Input(shape=(None, ))

        embedding_layer = Embedding(input_dim=self.params_json["total_labels_n"], output_dim=decoder_params["embedding_dim"])(sequence_input_layer)
        for layer_n in range(self.decoder_layers_n):
            
            if layer_n == (self.decoder_layers_n - 1):
                break
            
            if layer_n == 0:

                if decoder_params["bidirectional"][layer_n]:
                    lstm_layer = Bidirectional(LSTM(units=decoder_params["units"][layer_n], 
                                                    kernel_initializer=self.weights_init, 
                                                    return_sequences=True))(embedding_layer, initial_state=[input_layer, input_layer])
                    
                else:
                    lstm_layer = LSTM(units=decoder_params["units"][layer_n], kernel_initializer=self.weights_init, 
                                      return_sequences=True)(embedding_layer, initial_state=[input_layer, input_layer])
            
            else:

                if decoder_params["bidirectional"][layer_n]:
                    lstm_layer = Bidirectional(LSTM(units=decoder_params["units"][layer_n], kernel_initializer=self.weights_init, return_sequences=True))(lstm_layer)
                    
                else:
                    lstm_layer = LSTM(units=decoder_params["units"][layer_n], kernel_initializer=self.weights_init, return_sequences=True)(lstm_layer)
    
            if not decoder_params["single_dropout"]:
                lstm_layer = Dropout(rate=decoder_params["dropout_rates"][layer_n])(lstm_layer)

        lstm_layer = LSTM(units=decoder_params["units"][layer_n], kernel_initializer=self.weights_init, return_sequences=True)(lstm_layer)    
        if decoder_params["single_dropout"]:
            lstm_layer = Dropout(rate=0.26)(lstm_layer)
        
        decoder_output_layer = Dense(units=self.params_json["total_labels_n"], activation="softmax")(lstm_layer)
        self.decoder = Model(inputs=[input_layer, sequence_input_layer], outputs=decoder_output_layer)

    def _build_model_(self):
        
        model_input_layer = Input(shape=self.params_json["input_shape"])
        sequence_layer = Input(shape=(None, ))
        model_output_layer = self.decoder([self.encoder(model_input_layer), sequence_layer])

        self.model = Model(inputs=[model_input_layer, sequence_layer], outputs=model_output_layer)
        self.model.compile(loss=SparseCategoricalCrossentropy(), optimizer="rmsprop")
    

    def train_model(self, train_images, train_sequences, epochs, batch_size):

        self.model.fit([train_images, train_sequences], train_sequences, epochs=epochs, batch_size=batch_size)
        model_weights_folder = os.path.join(self.params_json["run_folder"], "model_weights.weights.h5")
        self.model.save_weights(filepath=model_weights_folder)

    
    def decoder_image(self, image, decoder_sequence_l):

        encoded_state_vector = self.encoder.predict(image)
        target_sequence = np.expand_dims(np.random.randint(0, self.params_json["total_words_n"]), axis=0)
        decoded_sequence = ""
        for _ in range(decoder_sequence_l):

            predicted_sample = self.decoder.predict([encoded_state_vector, target_sequence])
            target_sequence = np.expand_dims(np.random.randint(0, self.params_json["total_words_n"]), axis=0)
            decoded_sequence.append(predicted_sample[0])
        
        return decoded_sequence