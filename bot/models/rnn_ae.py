import numpy as np
import matplotlib.pyplot as plt
import json as js
import os 

from tensorflow.keras.layers import Input, Dense, Activation, LayerNormalization, Dropout
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, GRU, Attention
from tensorflow.keras.losses import CategoricalCrossentropy
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

        
        self._save_params_(filepath=params_path)
        

    def _save_params_(self, filepath):
        
        with open(filepath, "w") as file:
            js.dump(self.params_json, file)
    
    def _load_params_(self, filepath):
        
        with open(filepath, "r") as json_file:
            self.params_json = js.load(json_file)
    
    def _build_encoder_(self):
         
        encoder_params = self.params_json["encoder_params"]

        self.encoder_input_layer = Input(shape=(None, ))
        embedding_layer = Embedding(input_dim=encoder_params["total_words_n"], output_dim=self.params_json["embedding_dim"])(self.encoder_input_layer)
        lstm_layer = embedding_layer

        for _ in range(encoder_params["lstm_params"]["layers_n"]):
            
            lstm_layer = LSTM(units=encoder_params["lstm_params"]["units"], return_sequences=True, kernel_initializer=self.weights_init)(lstm_layer)
            lstm_layer = LayerNormalization(epsilon=0.01)(lstm_layer)
            lstm_layer = Dropout(rate=encoder_params["lstm_params"]["dropout_rate"])(lstm_layer)
        
        self.encoder = Model(inputs=self.encoder_input_layer, outputs=lstm_layer)
    
    def _build_decoder_(self):

        decoder_params = self.params_json["decoder_params"]
        
        input_layer = Input(shape=(None, ))
        embedding_layer = Embedding(input_dim=decoder_params["total_words_n"], output_dim=self.params_json["embedding_dim"])(input_layer)

        lstm_layer = LSTM(units=decoder_params["lstm_params"]["units"], return_sequences=True)(embedding_layer)
        input_layer_1 = Input(shape=lstm_layer.shape[1:])
        lstm_layer = Attention()([lstm_layer, input_layer_1])

        print(lstm_layer.shape)

        for _ in range(0, decoder_params["lstm_params"]["layers_n"] - 1):

            lstm_layer = LSTM(units=decoder_params["lstm_params"]["units"], return_sequences=True, kernel_initializer=self.weights_init)(lstm_layer)
            lstm_layer = LayerNormalization(epsilon=0.01)(lstm_layer)
            lstm_layer = Dropout(rate=decoder_params["lstm_params"]["dropout_rate"])(lstm_layer)
        
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
    
    
    
    def train_model(self, encoder_train_tensor, decoder_train_tensor, decoder_train_labels, batch_size, epochs):
        
        for _ in range(epochs):

            random_idx = np.random.randint(0, encoder_train_tensor.shape[0], batch_size)
            
            encoder_input_samples = encoder_train_tensor[random_idx]
            decoder_input_samples = decoder_train_tensor[random_idx]

            decoder_input_labels = decoder_train_labels[random_idx]
            decoder_input_labels = to_categorical(decoder_input_labels, num_classes=self.params_json["decoder_params"]["total_words_n"])

            for (decoder_sub_samples, decoder_sub_labels) in zip(decoder_input_samples, decoder_input_labels):

                self.model.train_on_batch([encoder_input_samples, decoder_sub_samples], decoder_sub_labels)
                

        

        
    
    def save_sample(self, input_encoder_sequence, filepath=None, input_sequence=None, sequence_lenght=100):

        encoded_sequence = self.encoder.predict(input_encoder_sequence)
        decoded_sequence = ""

        if input_sequence is None:
            input_sequence = np.zeros((1, self.params_json["decoder_params"]["total_labels_n"]))
        
        
        for _ in range(sequence_lenght):

            decoded_word_logits = self.decoder.predict([input_sequence, encoded_sequence])
            decoded_label = np.argmax(decoded_word_logits)
            decoded_word = self.encoder_tokenizer.index_word[decoded_label]
        
            decoded_sequence += decoded_word
        
        decoded_sequence.repleace("---", "\n")
        if filepath is None:
            return decoded_sequence

        if filepath is not None:
            
            with open(filepath, "w") as file:
                file.write(decoded_sequence)
    
        

if __name__ == "__main__":

    encoder_input_data = np.random.randint(0, 1000, (100, 30))
    decoder_input_data = np.random.randint(0, 1000, (100, 100))
    
    decoder_data_tensor = []
    decoder_data_labels = []
    sample_lenght = 20

    for (sample_number, sample) in enumerate(decoder_input_data):
        
        sample_tensor = []
        sample_labels = []    
        for i in range(sample.shape[0] - sample_lenght):
            
            sample_tensor.append(sample[i: i + sample_lenght])
            sample_labels.append(sample[i + sample_lenght])

        sample_tensor = np.asarray(sample_tensor, dtype="int")
        sample_labels = np.asarray(sample_labels, dtype="int")

        decoder_data_tensor.append(sample_tensor)
        decoder_data_labels.append(sample_labels)
    
    decoder_data_tensor = np.asarray(decoder_data_tensor, dtype="int")
    decoder_data_labels = np.asarray(decoder_data_labels, dtype="int")
    print(decoder_data_tensor.shape, decoder_data_labels.shape)
        
    len_labels = np.ones(encoder_input_data.shape[0]) * 30


    encoder_total_words = np.argmax(encoder_input_data)
    decoder_total_words = np.argmax(decoder_input_data)

    params_json = {
        "embedding_dim": 1000,
        "weights_init": {
            "mean": 0.0,
            "stddev": 1.0
        },
        "encoder_params": {
            "total_words_n": int(encoder_total_words),
            "lstm_params": {
                "layers_n": 3,
                "units": 215,
                "dropout_rate": 0.26
            }
        },
        "decoder_params": {
            "total_words_n": int(decoder_total_words),
            "lstm_params": {
                "layers_n": 3,
                "units": 215,
                "dropout_rate": 0.26
            },
        }
    }
    rnn_ae = RNN_AE(params_json=params_json)
    print(rnn_ae.encoder.summary())
    print(rnn_ae.decoder.summary())

    rnn_ae.train_model(encoder_train_tensor=encoder_input_data, 
                        decoder_train_tensor=decoder_data_tensor,
                        decoder_train_labels=decoder_data_labels, 
                        batch_size=80, epochs=100)



        

        
    



            