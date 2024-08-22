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


class Tocotron2:

    def __init__(self, params_json=None, params_path=None) -> None:
        
        self.params_json = params_json
        if self.params_json is None:
            self._load_params_(filepath=params_path)

        self._build_encoder_()
        self._build_decoder_()
        self._build_model_()
        self._save_params_(filepath=params_path)
        
    def _load_params_(self, filepath):
        
        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\Tacotron2.json"

        with open(filepath, "r") as json_file:
            self.params_json = js.load(json_file)
    
    def _save_params_(self, filepath):

        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\Tacotron2.json"

        with open(filepath, "w") as json_file:
            js.dump(self.params_json, json_file)
    
    def _build_encoder_(self):
        
        encoder_params = self.params_json["encoder_params"]
        input_layer = Input(shape=(None, ), name=f"EncoderInputLayer")
        embedding_layer = Embedding(input_dim=encoder_params["total_labels_n"], output_dim=encoder_params["embedding_dim"], name="EncoderEmbeddingLayer")(input_layer)

        conv_layer = embedding_layer
        conv_params = encoder_params["conv_params"]
        for layer_n in range(len(conv_params["filters"])):

            conv_layer = Conv1D(filters=conv_params["filters"][layer_n], kernel_size=conv_params["kernel_size"][layer_n],
                                    strides=conv_params["strides"][layer_n], padding="same", 
                                    name=f"EncoderConv1DLayer{layer_n}")(conv_layer)
                
            conv_layer = Activation(conv_params["activations"][layer_n], name=f"EncoderActivationLayer{layer_n}")(conv_layer)
            conv_layer = Dropout(rate=conv_params["dropout_rates"], name=f"EncoderDropoutLayer{layer_n}")(conv_layer)
            
        output_layer = Bidirectional(LSTM(units=encoder_params["lstm_units"], return_sequences=True, name=f"EncoderBidirectionalLayer{layer_n}"))(conv_layer)
        self.encoder = Model(inputs=input_layer, outputs=output_layer)
    
    def _build_decoder_(self):
        
        decoder_params = self.params_json["decoder_params"]
        input_layer = Input(shape=(None, ), name="DecoderInputLayer")
        embedding_layer = Embedding(input_dim=decoder_params["total_labels_n"], output_dim=decoder_params["embedding_dim"], name="DecoderEmbeddingLayer")(input_layer)
        
        lstm_layer = Bidirectional(LSTM(units=decoder_params["lstm_params"]["units"], return_sequences=True, name="DecoderInputLSTMLayer"))(embedding_layer)
        encoder_output_layer = Input(shape=lstm_layer.shape[1:], name="InputFromEncoderLayer")
        
        attention_layer = Attention(name="DecoderAttentionLayer")([encoder_output_layer, lstm_layer])
        lstm_params = decoder_params["lstm_params"]
        for layer_n in range(1, lstm_params["layers_n"]):

            if layer_n == 1:
                lstm_layer = LSTM(units=lstm_params["units"], 
                                return_sequences=True, 
                                name=f"DecoderLSTMLayer{layer_n}")(attention_layer)
            
            else:
                lstm_layer = LSTM(units=lstm_params["units"], 
                                return_sequences=True, 
                                name=f"DecoderLSTMLayer{layer_n}")(lstm_layer)
                
            lstm_layer = Dropout(rate=lstm_params["dropout_rates"], name=f"DecoderLSTMDropoutLayer{layer_n}")(lstm_layer)
        
        lstm2dense_layer = LSTM(units=lstm_params["units"], name="DecoderLSTMLayerLast")(lstm_layer)
        dense_layer = Dense(units=decoder_params["dense_params"]["units"], activation=decoder_params["dense_params"]["activation"], name="DecoderDenseLayer")(lstm2dense_layer)
        
        conv_layer = embedding_layer
        conv_params = decoder_params["conv_params"]
        for layer_n in range(len(conv_params["filters"])):

            conv_layer = Conv1D(filters=conv_params["filters"][layer_n], kernel_size=conv_params["kernel_size"][layer_n],
                                strides=conv_params["strides"][layer_n], padding="same", name=f"DecoderConv1DLayer{layer_n}")(conv_layer)
        
            
            conv_layer = Activation(conv_params["activations"][layer_n])(conv_layer)
            conv_layer = Dropout(rate=conv_params["dropout_rates"], name=f"DecoderConv1DDropoutLayer{layer_n}")(conv_layer)
        
        
        dense2conv_layer = RepeatVector(n=decoder_params["max_sequence_lenght"])(dense_layer)
        mel_sequence_output = Concatenate(axis=1, name="DecoderOutputLayer")([conv_layer, dense2conv_layer])
        mel_sequence_output = Activation(conv_params["output_activation"], name="DecoderOutputActivation")(conv_layer)

        self.decoder = Model(inputs=[input_layer, encoder_output_layer], outputs=mel_sequence_output)
        
    def _build_model_(self):

        encoder_input_layer = Input(shape=(None, ))
        decoder_input_layer = Input(shape=(None, ))

        encoder_forward = self.encoder(encoder_input_layer)
        decoder_forward = self.decoder([decoder_input_layer, encoder_forward])

        wavenet_params = self.params_json["wavenet_params"]
        model_output_layer = WaveNetLayer(filters=wavenet_params["filters"], kernel_size=wavenet_params["kernel_size"],
                                     output_dim=wavenet_params["output_dim"])(decoder_forward)
        
        self.model = Model(inputs=[encoder_input_layer, decoder_input_layer], outputs=model_output_layer)
        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    
    def train_model(self, train_texts_sequences, train_voice_sequences, epochs, batch_size):
        
        weights_folder = "c:\\Users\\1\\Desktop\\models_save\\Tacotron2"
        if not os.path.exists(weights_folder):
            os.mkdir(weights_folder)
        
        weights_path = os.path.join(weights_folder, "weights.weights.h5")
        self.model.fit([train_texts_sequences, train_voice_sequences], 
                       train_voice_sequences, epochs=epochs, 
                       batch_size=batch_size, shuffle=True)
        self.model.save_weigths(filepath=weights_path)
    
    def decode_sequence(self, text_sequences, number_of_sequences):
        
        sequence_lenght = self.params_json["wavenet_params"]["max_seq_len"]
        max_audio_aplitude = self.params_json["wavenet_params"]["output_dim"]

        encoder_output = self.encoder.predict(text_sequences)
        audio_sample = np.random.randint(0, max_audio_aplitude, (1, sequence_lenght))
        pred_sequences = [] + audio_sample.tolist()
        pred_mel_spects = []

        for _ in range(number_of_sequences):

            pred_mel = self.decoder.predict([encoder_output, audio_sample])
            pred_audio_sample = self.wavenet_model.predict(pred_mel)
            pred_audio_sample = np.asarray([np.argmax(sample) for sample in pred_audio_sample], dtype="int")

            audio_sample = pred_audio_sample
            pred_sequences += pred_audio_sample.tolist()
            pred_mel_spects.append(pred_mel)
            
        return pred_sequences
        
        


    
   

    


        
        
        