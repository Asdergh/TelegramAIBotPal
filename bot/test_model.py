import numpy as np
import matplotlib.pyplot as plt
import random as rd
import json as js
import os
import librosa as libos
from librosa import beat as bt
from librosa.feature import melspectrogram
from layers import WaveNetLayer

from tensorflow.keras.layers import Attention, LSTM, Input, Embedding, Dense, RepeatVector, Dropout, Concatenate
from tensorflow.keras.layers import Conv2D, Conv1D, Reshape, Activation, Permute, Multiply, Bidirectional
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical



def build_encoder(params_json):
    

    input_layer = Input(shape=(None, ), name=f"EncoderInputLayer")
    embedding_layer = Embedding(input_dim=params_json["total_labels_n"], output_dim=params_json["embedding_dim"], name="EncoderEmbeddingLayer")(input_layer)

    conv_layer = embedding_layer
    conv_params = params_json["conv_params"]
    for layer_n in range(len(conv_params["filters"])):

        conv_layer = Conv1D(filters=conv_params["filters"][layer_n], kernel_size=conv_params["kernel_size"][layer_n],
                            strides=conv_params["strides"][layer_n], padding="same", 
                            name=f"EncoderConv1DLayer{layer_n}")(conv_layer)
        
        conv_layer = Activation(conv_params["activations"][layer_n], name=f"EncoderActivationLayer{layer_n}")(conv_layer)
        conv_layer = Dropout(rate=conv_params["dropout_rates"], name=f"EncoderDropoutLayer{layer_n}")(conv_layer)
    
    output_layer = Bidirectional(LSTM(units=params_json["lstm_units"], return_sequences=True, name=f"EncoderBidirectionalLayer{layer_n}"))(conv_layer)
    encoder = Model(inputs=input_layer, outputs=output_layer)

    return encoder

def build_decoder(params_json):

    input_layer = Input(shape=(None, ), name="DecoderInputLayer")
    

    embedding_layer = Embedding(input_dim=params_json["total_labels_n"], output_dim=params_json["embedding_dim"], name="DecoderEmbeddingLayer")(input_layer)
    lstm_layer = Bidirectional(LSTM(units=params_json["lstm_params"]["units"], return_sequences=True, name="DecoderInputLSTMLayer"))(embedding_layer)
    encoder_output_layer = Input(shape=lstm_layer.shape[1:], name="InputFromEncoderLayer")
    
    attention_layer = Attention(name="DecoderAttentionLayer")([encoder_output_layer, lstm_layer])
    lstm_params = params_json["lstm_params"]
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
    dense_layer = Dense(units=params_json["dense_params"]["units"], activation=params_json["dense_params"]["activation"], name="DecoderDenseLayer")(lstm2dense_layer)
    stop_condition_output = Dense(units=1, activation=params_json["dense_params"]["activation"], name="DecoderFirstOutputLayer")(dense_layer)
    
    conv_layer = embedding_layer
    conv_params = params_json["conv_params"]
    for layer_n in range(len(conv_params["filters"])):

        conv_layer = Conv1D(filters=conv_params["filters"][layer_n], kernel_size=conv_params["kernel_size"][layer_n],
                            strides=conv_params["strides"][layer_n], padding="same", name=f"DecoderConv1DLayer{layer_n}")(conv_layer)
       
        
        conv_layer = Activation(conv_params["activations"][layer_n])(conv_layer)
        conv_layer = Dropout(rate=conv_params["dropout_rates"], name=f"DecoderConv1DDropoutLayer{layer_n}")(conv_layer)
    
    
    dense2conv_layer = RepeatVector(n=params_json["max_sequence_lenght"])(dense_layer)
    mel_sequence_output = Concatenate(axis=1, name="DecoderSecondOutputLayer")([conv_layer, dense2conv_layer])
    mel_sequence_output = Activation(conv_params["output_activation"], name="DecoderSecondOutputActivation")(conv_layer)

    decoder = Model(inputs=[input_layer, encoder_output_layer], outputs=[stop_condition_output, mel_sequence_output])
    return decoder



params_json = {
    "run_folder": "c:\\Users\\1\\Desktop\\PersonalFriendProject\\models_save\\Tacotron2_save",
    "params_path": "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\Tacotron2.json",
    "encoder_params": {

        "total_labels_n": 1000,
        "embedding_dim": 100,
        "max_sequence_lenght": 1000,

        "conv_params": {

            "filters": [32, 32, 128, 128],
            "kernel_size": [3, 3, 3, 3],
            "strides": [1, 2, 2, 1],
            "dropout_rates": 0.26,
            "activations": ["linear", "linear", "linear", "linear"]
        },
        "lstm_units": 215
    },

    "decoder_params": {

        "total_labels_n": 10000,
        "max_sequence_lenght": 10000,
        "embedding_dim": 100,

        "dense_params": {
            "units": 215,
            "dropout_rates": 0.26,
            "activation": "linear"
        },

        "lstm_params": {
            "units": 215,
            "dropout_rates": 0.26,
            "layers_n": 3
        },

        "conv_params": {
            "filters": [32, 32, 128, 215],
            "kernel_size": [3, 3, 3, 3],
            "strides": [1, 2, 2, 1],
            "dropout_rates": 0.26,
            "activations": ["linear", "linear", "linear", "linear"],
            "output_activation": "tanh"
        }
    }
}


test_data = np.random.randint(0, 10000, (100, 100))
test_labels = np.random.randint(0, 10000, 100)
test_labels = [[0.0 if index != label else 1.0 for index in range(np.max(test_data))] for label in test_labels]
test_labels = np.asarray(test_labels, dtype="float64")


input_layer = Input(shape=(None, ))

embedding_layer = Embedding(input_dim=np.max(test_data) + 1, output_dim=100)(input_layer)
conv_layer = Conv1D(filters=128, kernel_size=3, strides=1, padding="same")(embedding_layer)
conv_layer = Conv1D(filters=215, kernel_size=3, strides=1, padding="same")(conv_layer)
conv_layer = Conv1D(filters=215, kernel_size=3, strides=1, padding="same")(conv_layer)
wave_net_layer = WaveNetLayer(filters=128, kernel_size=3, output_dim=np.max(test_data), max_sequence_lenght=12)(conv_layer)

model = Model(inputs=input_layer, outputs=wave_net_layer)
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

model_preds = model.predict(test_data)
print(model_preds.shape, model_preds)

    





# encoder_test_tensor = np.random.randint(0, 1000, (100, 30))
# decoder_test_tensor = np.random.randint(0, 1000, (100, 1000))


# encoder_input_layer = Input(shape=(None, ))
# decoder_input_layer = Input(shape=(None, ))

# encoder = build_encoder(params_json=params_json["encoder_params"])
# encoder_output = encoder.predict(encoder_test_tensor)
# print(encoder_output.shape)
# decoder = build_decoder(params_json["decoder_params"])
# model_output = decoder([decoder_input_layer, encoder(encoder_input_layer)])
# encoder_output = encoder.predict(encoder_test_tensor)
# decoder_output = decoder.predict([decoder_test_tensor, encoder_output])

# print(encoder_output.shape, decoder_output[1].shape)
# print(encoder_output, decoder_output[1])

# model = Model(inputs=[encoder_input_layer, decoder_input_layer], outputs=model_output)
# print(model.summary())

# plt.style.use("dark_background")
# fig, axis = plt.subplots(nrows=5, ncols=5)

# sample_number = 0
# for i in range(5):
#     for j in range(5):
        
#         curent_sample = decoder_output[1][sample_number]
#         axis[i, j].imshow(curent_sample, cmap="inferno")
#         sample_number += 1

# plt.show()



