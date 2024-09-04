import numpy as np
import random as rd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Embedding, LSTM, Conv1D, Dropout, Attention, LayerNormalization
from tensorflow.keras import Model, Sequential

input_data = np.random.randint(0, 10000, (100, 30))
total_words = np.max(input_data)
test_dataset = None

params_json = {
    "encoder_params": {        
        "total_words_n":    1000,
        "embedding_size":   100,
        "layers_n":         3,
        "units":            256,
        "activation":       None,
        "dropout_rate":    0.34
    },

    "decoder_params": {
        "prenet_params": {
            "epsilon": 0.001,
            "dropout_rate": 0.26
        },
        "filters":          [32, 32, 64, 128                ],
        "kernel_size":      [5, 5, 5, 5                     ],
        "padding":          ["same", "same", "same", "same" ],
        "strides":          [1, 1, 1, 1                     ],
        "dropout_rates":    [0.26, 0.26, 0.26, 0.26         ],
        "normalization_epsilon": 0.001
    }
}

def build_encoder(params_json):

    encoder_params = params_json["encoder_params"]

    input_layer = Input(shape=(None, ))
    embedding_layer = Embedding(input_dim=encoder_params["total_words_n"], output_dim=encoder_params["embedding_size"])(input_layer)
    lstm_layer = embedding_layer

    for _ in range(encoder_params["layers_n"]):

        lstm_layer = LSTM(units=encoder_params["units"], return_sequences=True)(lstm_layer)
        lstm_layer = Dropout(rate=encoder_params["dropout_rate"])(lstm_layer)
    
    encoder = Model(inputs=input_layer, outputs=lstm_layer)
    return encoder

def build_decoder(params_json, encoder_output):

    decoder_params = params_json["decoder_params"]

    input_layer = Input(shape=(None, decoder_params["filters"][-1]))
    prenet_layer = Sequential(layers=[
        LayerNormalization(epsilon=decoder_params["prenet_params"]["epsilon"]),
        Dropout(rate=decoder_params["prenet_params"]["dropout_rate"])])(input_layer)

    lstm_layer = LSTM(units=params_json["encoder_params"]["units"], return_sequences=True)(prenet_layer)
    att_layer = Attention()([lstm_layer, encoder_output])

    conv_layer = att_layer
    for layer_n in range(len(decoder_params["filters"])):

        conv_layer = Conv1D(filters=decoder_params["filters"][layer_n], kernel_size=decoder_params["kernel_size"][layer_n],
                            padding=decoder_params["padding"][layer_n], strides=decoder_params["strides"][layer_n])(conv_layer)
        
        conv_layer = LayerNormalization(epsilon=decoder_params["normalization_epsilon"])(conv_layer)
        conv_layer = Dropout(rate=decoder_params["dropout_rates"][layer_n])(conv_layer)
    
    decoder = Model(inputs=[input_layer, encoder_output], outputs=conv_layer)
    return decoder

input_layer = Input(shape=(None, ))
decoder_input_layer = Input(shape=(None, 128, ))

encoder = build_encoder(params_json=params_json)
encoder_output = encoder(input_layer)
decoder = build_decoder(params_json=params_json, encoder_output=encoder_output)
decoder_output = decoder([decoder_input_layer, encoder_output])

model = Model(inputs=[input_layer, decoder_input_layer], outputs=decoder_output)
model.compile(optimizer="adam", loss="mean_squared_error")


test_input = np.random.randint(0, 1000, (100, 100))
decoder_input = np.zeros((100, 100, 128))

model_output = model.fit([test_input, decoder_input], decoder_input, epochs=10, batch_size=32)

plt.style.use("dark_background")
fig, axis = plt.subplots()

new_test = np.zeros((1, 100, 128))
encoded_output = encoder.predict(np.expand_dims(test_input[0], axis=0))
decoded_output = decoder.predict([new_test, encoded_output])

axis.imshow(decoded_output[0], cmap="inferno")
plt.show()







    
    
    

    
