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


class RNN:

    def __init__(self, rnn_params=None, filepath=None) -> None:
        

        

        self.rnn_params = rnn_params
        if filepath is None:
                filepath ="C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\RNN.json"

        if self.rnn_params is None:
            self._load_params_(filepath=filepath)

        self.tokenizer = Tokenizer()
        self.max_sequence_lenght = self.rnn_params["max_sequence_lenght"]
        self.model_rnn_layers_n = len(self.rnn_params["rnn_params"]["units"])
        self.model_dense_layers_n = len(self.rnn_params["dense_params"]["units"])
        self.weights_init = RandomNormal(mean=0.12, stddev=0.22)
        
        if self.rnn_params["model_type"] == "generator":
            self._load_text_data_()

        elif self.rnn_params["model_type"] == "classificator":
            
            if "classes_discription" in self.rnn_params.keys():
                self.total_cll_n = len(self.rnn_params["classes_discription"])
                self.total_labels_n = self.rnn_params["total_labels_n"]
            
            else:
                raise ValueError("!!can't find need params in params_json: [classes_discription]!!")
            
        self._build_model_()
        self._save_params_(filepath=filepath)
        self.learning_history = []
        self.epoch_iterator = 0
    
    def _build_model_(self):

        input_layer = Input(shape=(None, ))
        embedding_layer = Embedding(input_dim=self.total_labels_n, output_dim=self.rnn_params["embedding_dim"])(input_layer)
        
        if self.model_rnn_layers_n != 1:

            for layer_n in range(self.model_rnn_layers_n):
                
                if layer_n == 0:
                    
                    if (self.rnn_params["rnn_params"]["layer_type"] == "lstm"):
                        rnn_layer = LSTM(units=self.rnn_params["rnn_params"]["units"][layer_n], 
                                    kernel_initializer=self.weights_init, 
                                    return_sequences=True)(embedding_layer)
                    
                    elif (self.rnn_params["rnn_params"]["layer_type"] == "gru"):
                        rnn_layer = GRU(units=self.rnn_params["rnn_params"]["units"][layer_n], 
                                    kernel_initializer=self.weights_init, 
                                    return_sequences=True)(embedding_layer)

            
                else:
                    
                    if (self.rnn_params["rnn_params"]["layer_type"] == "lstm"):
                        rnn_layer = LSTM(units=self.rnn_params["rnn_params"]["units"][layer_n], 
                                        kernel_initializer=self.weights_init,  
                                        return_sequences=True)(rnn_layer)
                    
                    elif (self.rnn_params["rnn_params"]["layer_type"] == "gru"):
                        rnn_layer = GRU(units=self.rnn_params["rnn_params"]["units"][layer_n], 
                                        kernel_initializer=self.weights_init,  
                                        return_sequences=True)(rnn_layer)



                if "activations" in self.rnn_params["rnn_params"].keys():
                    rnn_layer = Activation(self.rnn_params["rnn_params"]["activations"][layer_n])(rnn_layer)

                if self.rnn_params["rnn_params"]["bidirectional_init"]:
                    
                    if (self.rnn_params["rnn_params"]["layer_type"] == "lstm"):
                        rnn_layer = Bidirectional(LSTM(units=self.rnn_params["rnn_params"]["units"][layer_n], return_sequences=True))(rnn_layer)

                    if (self.rnn_params["rnn_params"]["layer_type"] == "gru"):
                        rnn_layer = Bidirectional(GRU(units=self.rnn_params["rnn_params"]["units"][layer_n], return_sequences=True))(rnn_layer)

            if (self.rnn_params["rnn_params"]["layer_type"] == "lstm"):
                rnn_layer = LSTM(self.rnn_params["rnn_params"]["units"][-1], kernel_initializer=self.weights_init)(rnn_layer)

            elif (self.rnn_params["rnn_params"]["layer_type"] == "gru"):
                rnn_layer = GRU(self.rnn_params["rnn_params"]["units"][-1], kernel_initializer=self.weights_init)(rnn_layer)

        else:
            
            if (self.rnn_params["rnn_params"]["layer_type"] == "lstm"):
                rnn_layer = LSTM(units=self.rnn_params["rnn_params"]["units"][0], 
                                kernel_initializer=self.weights_init,
                                return_sequences=True)(embedding_layer)

            elif (self.rnn_params["rnn_params"]["layer_type"] == "gru"):
                rnn_layer = GRU(units=self.rnn_params["rnn_params"]["units"][0], 
                                kernel_initializer=self.weights_init, 
                                return_sequences=True)(embedding_layer)
            
            if (self.rnn_params["rnn_params"]["layer_type"] == "lstm"):
                    rnn_layer = Bidirectional(LSTM(units=self.rnn_params["rnn_params"]["units"][0]))(rnn_layer)
                        
            elif (self.rnn_params["rnn_params"]["layer_type"] == "gru"):
                    rnn_layer = Bidirectional(GRU(units=self.rnn_params["rnn_params"]["units"][0]))(rnn_layer)
            
            if "activations" in self.rnn_params["rnn_params"].keys():
                    rnn_layer = Activation(self.rnn_params["rnn_params"]["activations"][0])(rnn_layer)
                
            
        rnn_layer = Dropout(rate=self.rnn_params["rnn_params"]["dropout_rate"])(rnn_layer)
        rnn_layer = LayerNormalization()(rnn_layer)
        dense_layer = rnn_layer

        if self.model_dense_layers_n != 1:
        
            for layer_n in range(self.model_dense_layers_n):

                dense_layer = Dense(units=self.rnn_params["dense_params"]["units"][layer_n], 
                                    activation=self.rnn_params["dense_params"]["activations"][layer_n])(dense_layer)

                dense_layer = Dropout(rate=self.rnn_params["dense_params"]["dropout_rates"][layer_n])(dense_layer)
            
            output_layer = Dense(units=self.total_cll_n, activation="softmax")(dense_layer)
        
        elif self.model_dense_layers_n == 1:

            dense_layer = Dense(units=self.rnn_params["dense_params"]["units"][0])(dense_layer)
            dense_layer = Dropout(rate=self.rnn_params["dense_params"]["dropout_rates"][0])(dense_layer)

            dense_layer = Dense(units=self.total_cll_n, activation="sigmoid")(dense_layer)
            output_layer = Dense(units=self.total_cll_n, activation="softmax")(dense_layer)

        elif self.model_dense_layers_n == 0:

            dense_layer = Dense(units=self.total_cll_n, activation="sigmoid")(dense_layer)
            output_layer = Dense(units=self.total_cll_n, activation="softmax")(dense_layer)
            
        
        self.model = Model(input_layer, output_layer)
        self.model.compile(loss="categorical_crossentropy", optimizer=RMSprop(self.rnn_params["learning_rate"]), metrics=["accuracy"])
    
    
    def _prepeare_data_(self, encoded_text):

        train_tensor = []
        train_labels = []


        for i in range(self.total_cll_n - self.max_sequence_lenght):

            train_tensor.append(encoded_text[i: (i + self.max_sequence_lenght)])
            train_labels.append(encoded_text[i + self.max_sequence_lenght])

        train_tensor = np.asarray(train_tensor)
        train_labels = np.asarray(train_labels)
        train_labels = to_categorical(train_labels, num_classes=self.total_cll_n)

        return (train_tensor, train_labels)
    
    def generate_msg(self, samples_number, seed_text=None, run_folder=None):
        
        if seed_text is None:
            
            random_idx = np.random.randint(0, self.train_tensor.shape[0])
            token_samples = self.train_tensor[random_idx]
            seed_text = self.tokenizer.sequences_to_texts([token_samples])[0]

        output_msg = ""
        for _ in range(samples_number):

            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = token_list[-self.max_sequence_lenght:]
            token_list = np.reshape(token_list, (1, self.max_sequence_lenght))
            
            word_logits = self.model.predict(token_list, verbose=0)
            word_label = np.argmax(word_logits)
            output_word = self.tokenizer.index_word[word_label] if word_label > 0 else ""

            output_msg += f" {output_word}"
            seed_text += f" {output_word}"
        
        if run_folder is None:
            return output_msg
        
        else:

            epoch_gen_samples = os.path.join(run_folder, f"EPOCH{self.epoch_iterator}_gen_smaples.txt")
            with open(epoch_gen_samples, "w", encoding="utf-8") as file:
                file.write(output_msg)
            
    
    def _load_text_data_(self):

        with open(self.rnn_params["msgs_logs"], "r", encoding="utf-8") as text_log:
            
            text_data = text_log.read()
            text_data.replace("\n", "|")
        
        self.tokenizer.fit_on_texts([text_data])
        self.total_labels_n = len(self.tokenizer.word_index) + 1
        self.rnn_params["total_labels_n"] = self.total_labels_n

        self.encoded_text = self.tokenizer.texts_to_sequences([text_data])[0]
        self.train_tensor, self.train_labels = self._prepeare_data_(encoded_text=self.encoded_text)


    def train_model(self, epochs, batch_size, train_tensor=None, train_labels=None, learning_dim=1000):
        
        if self.rnn_params["model_type"] == "generator":
            
            random_idx = np.random.randint(0, self.train_tensor.shape[0], learning_dim)
            train_tensor = self.train_tensor[random_idx]
            train_labels = self.train_labels[random_idx]
        
        elif (self.rnn_params["model_type"] == "classificator" and (train_tensor is None or train_labels is None)):
            raise ValueError("!!can't find need arguments: [train_tensor, train_labels] for 'classication' model type!!")
        
        random_idx = np.random.randint(0, train_tensor.shape[0], learning_dim)
        train_random_samples = train_tensor[random_idx]
        train_random_labels = train_labels[random_idx]
        self.learnin_history = self.model.fit(train_random_samples, train_random_labels, epochs=epochs, batch_size=batch_size, shuffle=True)

        if not os.path.exists(self.rnn_params["run_folder"]):
            os.mkdir(self.rnn_params["run_folder"])

        model_weights_path = os.path.join(self.rnn_params["run_folder"], "model_weights.weights.h5")
        self.model.save_weights(filepath=model_weights_path)
    
    def _save_params_(self, filepath):

        with open(filepath, "w") as file:
            js.dump(self.rnn_params, file)
    
    def _load_params_(self, filepath):
        
        with open(filepath, "r") as json_file:
            self.rnn_params = js.load(json_file)
    
    def load_weights(self, filepath=None):

        if filepath is None:
            filepath = "c:\\Users\\1\\Desktop\\models_save\\RNN_classification_save\\prototype_2\\model_weights.weights.h5"

        self.model.load_weights(filepath)
    
    def expand_tokenizer(self, new_words):
        self.tokenizer.fit_on_texts(new_words)