import numpy as np
import matplotlib.pyplot as plt
import json as js
import random as rd
import os 

from tensorflow.keras.layers import Input, Dense, Activation, LayerNormalization, Dropout
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, GRU, Attention
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical


class RNN_AE:

    def __init__(self, params_json=None, params_path=None) -> None:
        
        self.params_json = params_json
        if params_path is None:
                params_path = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\RNN_AE.json"

        if self.params_json is None:
            self._load_params_(filepath=params_path)

        self.weights_init = RandomNormal(mean=self.params_json["weights_init"]["mean"], stddev=self.params_json["weights_init"]["stddev"])
        self.encoder_tokenizer = Tokenizer()
        self.decoder_tokenizer = Tokenizer()

        self._build_encoder_()
        self._build_decoder_()
        self._build_model_()

        self.model_losses = []

        
        self._save_params_(filepath=params_path)
        
    
    def load_tokenizers(self, encoder_tokenizer, decoder_tokenizer):
        
        model_wocab_folder = os.path.join(self.params_json["run_folder"], "model_wocab")
        if not os.path.exists(model_wocab_folder):
            os.mkdir(model_wocab_folder)
        
        encoder_wocab_path = os.path.join(model_wocab_folder, "encoder_wocab.json")
        decoder_wocab_path = os.path.join(model_wocab_folder, "decoder_wocab.json")
        
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        
        paths = [encoder_wocab_path, decoder_wocab_path]
        contents = [self.encoder_tokenizer.word_index, self.decoder_tokenizer.word_index]
        
        for (content, path) in zip(contents, paths):
            
            with open(path, "w") as json_file:
                js.dump(content, json_file)
    
    def expand_tokenizer(self, new_words, model_type="decoder"):

        wocab_folder = os.path.join(self.params_json["run_folder"], "model_wocab")
        wocab_path = os.path.join(wocab_folder, f"{model_type}_wocab.json")
        with open(wocab_path, "r") as json_file:
            word_index = js.load(json_file)
        
        for word in new_words:

            if word not in word_index.keys():
                word_index[word] = word_index.values()[-1] + 1
        
        if model_type == "decoder":
            self.decoder_tokenizer.word_index = word_index
        
        elif model_type == "encoder":
            self.decoder_tokenizer.word_index = word_index
        
        else:
            raise ValueError(f"wrong modeltpye: [{model_type}]!!!")
        

    
    def _save_params_(self, filepath):
        
        with open(filepath, "w") as file:
            js.dump(self.params_json, file)
    
    def _load_params_(self, filepath):
        
        with open(filepath, "r") as json_file:
            self.params_json = js.load(json_file)
    
    def _build_encoder_(self):
         
        encoder_params = self.params_json["encoder_params"]

        self.encoder_input_layer = Input(shape=(None, ), 
                                        name="EncoderInputLayer")
        embedding_layer = Embedding(input_dim=encoder_params["total_words_n"], 
                                    output_dim=self.params_json["embedding_dim"],
                                    name="EmbeddingLayer")(self.encoder_input_layer)
        lstm_layer = embedding_layer

        for layer_n in range(encoder_params["lstm_params"]["layers_n"]):
            
            lstm_layer = LSTM(units=encoder_params["lstm_params"]["units"], 
                            return_sequences=True, 
                            kernel_initializer=self.weights_init, 
                            name=f"EncoderLSTMLayer{layer_n}")(lstm_layer)
            
            lstm_layer = LayerNormalization(epsilon=0.01, 
                                name=f"LayerNormalization{layer_n}")(lstm_layer)
            lstm_layer = Dropout(rate=encoder_params["lstm_params"]["dropout_rate"], 
                                name=f"LSTMDropoutRate{layer_n}")(lstm_layer)
        
        self.encoder = Model(inputs=self.encoder_input_layer, outputs=lstm_layer)
    
    def _build_decoder_(self):

        decoder_params = self.params_json["decoder_params"]
        
        input_layer = Input(shape=(None, ),  name="DecoderInputLayer")
        embedding_layer = Embedding(input_dim=decoder_params["total_words_n"], 
                                    output_dim=self.params_json["embedding_dim"],
                                    name="DecoderEmbeddingLayer")(input_layer)

        lstm_layer = LSTM(units=decoder_params["lstm_params"]["units"], return_sequences=True,
                        name="LSTMInputLayer")(embedding_layer)
        input_layer_1 = Input(shape=(lstm_layer.shape[1:]), name="DecoderSequenceInputLayer")
        lstm_layer = Attention()([lstm_layer, input_layer_1])

        for layer_n in range(0, decoder_params["lstm_params"]["layers_n"] - 1):

            lstm_layer = LSTM(units=decoder_params["lstm_params"]["units"], 
                            return_sequences=True, 
                            kernel_initializer=self.weights_init,
                            name=f"DecoderLSTMLayer{layer_n}")(lstm_layer)
            
            lstm_layer = LayerNormalization(epsilon=0.01, 
                                            name=f"DecoderNormalizationLayer{layer_n}")(lstm_layer)
            lstm_layer = Dropout(rate=decoder_params["lstm_params"]["dropout_rate"], 
                                 name=f"DecoderDropoutLayer{layer_n}")(lstm_layer)
        
        lstm_layer = LSTM(units=decoder_params["lstm_params"]["units"])(lstm_layer)
        probability_output_layer = Dense(units=decoder_params["total_words_n"], activation="softmax")(lstm_layer)
        self.decoder = Model(inputs=[input_layer, input_layer_1], outputs=probability_output_layer)

    def _build_model_(self):

        encoder_input_layer = Input(shape=(None, ))
        decoder_input_layer = Input(shape=(None, ))

        encoder_forward = self.encoder(encoder_input_layer)
        decoder_forward = self.decoder([decoder_input_layer, encoder_forward])

        self.model = Model(inputs=[encoder_input_layer, decoder_input_layer], outputs=decoder_forward)
        self.model.compile(loss=CategoricalCrossentropy(), optimizer=RMSprop(learning_rate=0.01))
    
    
    
    def train_model(self, encoder_train_tensor, 
                    decoder_train_tensor, decoder_train_labels, 
                    batch_size, epochs):

        weights_folder = os.path.join(self.params_json["run_folder"], "model_weights")
        if not os.path.exists(weights_folder):
            os.mkdir(weights_folder)

        weights_path = os.path.join(weights_folder, "weights.weights.h5")
        self.model.fit([encoder_train_tensor, decoder_train_tensor], 
                       decoder_train_labels, 
                       batch_size=batch_size, epochs=epochs)
        self.model.save_weights(filepath=weights_path)
    
    def generate_sequence(self, input_question, sequence_len, target_sequence_len):
        
        
        
    
        output_text = "["
    
        en_in = self.encoder_tokenizer.texts_to_sequences([input_question])[0]
        en_in = np.asarray(en_in)
        en_in = np.expand_dims(en_in, axis=0)
        en_out = self.encoder.predict(en_in)

        dec_tar_seq = [np.random.randint(0, self.params_json["decoder_params"]["total_words_n"]) for _ in range(target_sequence_len + 1)]
        for _ in range(sequence_len):
            
            
            dec_tar_array = np.asarray(dec_tar_seq[-target_sequence_len: ])
            dec_tar_array = np.expand_dims(dec_tar_array, axis=0)

            dec_logits = self.decoder.predict([dec_tar_array, en_out])
            dec_label = np.argmax(dec_logits)
            dec_word = self.decoder_tokenizer.index_word[dec_label]

            output_text += " " + dec_word
            dec_tar_seq.append(dec_label)
            rd.shuffle(dec_tar_seq)
        
        return output_text
            
    def load_weights(self, filepath=None):

        if filepath is None:
            filepath = "c:\\Users\\1\\Desktop\\models_save\\RNN_AE_save\\model_weights\\weights.weights.h5"
        self.model.load_weights(filepath=filepath)
    

        

            

             

            
        
    
    

        
        
        
    
        




        

        
    



            