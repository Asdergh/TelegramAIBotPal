import numpy as np
import matplotlib.pyplot as plt
import json as js
import os
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense, Activation, Reshape, LayerNormalization, BatchNormalization
from tensorflow.keras.layers import LSTM, GRU, Masking, Bidirectional, Dropout, Conv2D, Conv2DTranspose, Flatten
from tensorflow.keras.layers import Concatenate, Lambda, Embedding, Multiply, Layer
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.random import categorical
from tensorflow import map_fn, expand_dims, convert_to_tensor
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import RandomNormal




class ConvSequence(Layer):

    def __init__(self, units, samples_n, sequence_l):

        super(ConvSequence, self).__init__()
        self.units = units
        self.samples_n = samples_n
        self.sequence_l = sequence_l
    
    def call(self, input_layer):
        

        sequence = input_layer[np.random.randint(0, input_layer.shape[0])]
        for _ in range(1, self.samples_n):
            
            random_idx = np.random.randint(0, input_layer.shape[0])
            random_coeff = np.random.normal(0, 1, sequence.shape[0])
            sample = input_layer[random_idx]
            sequence += sample * random_coeff
        
        lstm_layer = LSTM(units=self.units, return_sequences=True)(sequence)
        return lstm_layer

    def compute_output_shape(self):
        return (None, None, self.units) 
    




class RNN:

    def __init__(self, rnn_params=None, filepath=None) -> None:
        

        

        self.rnn_params = rnn_params
        if filepath is None:
                filepath ="C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\RNN.json"

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
            filepath = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_save\\RNN_classification_save\\prototype_2\\model_weights.weights.h5"

        self.model.load_weights(filepath)
    
    def expand_tokenizer(self, new_words):
        self.tokenizer.fit_on_texts(new_words)







class GAN:

    def __init__(self, params_json, params_filepath=None, load_params=False) -> None:
        
        
        self.params_json = params_json
        if load_params:
            self.load_params(filepath=params_filepath)
        
        self.dis_conv_layers_n = len(self.params_json["dis_conv_filters"])
        self.dis_dense_layers_n = len(self.params_json["dis_dense_units"])
        self.gen_conv_layer = len(self.params_json["gen_conv_filters"])
        if self.params_json["weights_init"]["init"]:
            self.weights_init = RandomNormal(mean=self.params_json["weights_init"]["mean"],stddev=self.params_json["weights_init"]["stddev"])


        self.gen_losses = []
        self.dis_losses = []
        self.epoch_iterator = 0

        self._build_discriminator_()
        self._build_generator_()
        self._build_model_()
        self.save_params()

    def save_params(self):

        filepath = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\GAN.json"
        with open(filepath, "w") as json_file:
            js.dump(self.params_json, json_file)

    def load_params(self, filepath):

        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\GAN.json"
        with open(filepath, "r") as json_file:
            self.params_json = js.laod(json_file)

    def _build_discriminator_(self):

        input_layer = Input(shape=self.params_json["input_shape"])
        conv_layer = input_layer
        
        for layer_n in range(self.dis_conv_layers_n):
            
            if self.params_json["weights_init"]["init"]:
                conv_layer = Conv2D(filters=self.params_json["dis_conv_filters"][layer_n], 
                                    kernel_size=self.params_json["dis_conv_kernel_size"][layer_n],
                                    padding="same", kernel_initializer=self.weights_init,
                                    strides=self.params_json["dis_conv_strides"][layer_n])(conv_layer)
            
            else:
                conv_layer = Conv2D(filters=self.params_json["dis_conv_filters"][layer_n], 
                                    kernel_size=self.params_json["dis_conv_kernel_size"][layer_n],
                                    padding="same", strides=self.params_json["dis_conv_strides"][layer_n])(conv_layer)

            conv_layer = Activation(self.params_json["dis_conv_activations"][layer_n])(conv_layer)
            conv_layer = Dropout(rate=self.params_json["dis_conv_dropout_rates"][layer_n])(conv_layer)
            conv_layer = BatchNormalization()(conv_layer)
        
        self.saved_shape = conv_layer.shape[1:]
        dense_layer = Flatten()(conv_layer)

        for layer_n in range(self.dis_dense_layers_n):

            dense_layer = Dense(self.params_json["dis_dense_units"][layer_n],
                                activation=self.params_json["dis_dense_activations"][layer_n])(dense_layer)
            
            dense_layer = Dropout(rate=self.params_json["dis_dense_dropout_rates"][layer_n])(dense_layer)
        
        output_layer = Dense(units=1, activation="sigmoid")(dense_layer)
        self.discriminator = Model(input_layer, output_layer)
    
    def set_trainable(self, model, value):

        model.trainable = value
        for layer in model.layers:
            layer.trainable = value

    def _build_generator_(self):

        input_layer = Input(shape=(self.params_json["noise_dim"], ))
        rec_layer = Dense(units=np.prod(self.saved_shape))(input_layer)
        rec_layer = Reshape(target_shape=self.saved_shape)(rec_layer)
        conv_layer = rec_layer

        for layer_n in range(self.gen_conv_layer):
            
            if self.params_json["weights_init"]["init"]:
                conv_layer = Conv2DTranspose(filters=self.params_json["gen_conv_filters"][layer_n], 
                                        kernel_size=self.params_json["gen_conv_kernel_size"][layer_n],
                                        padding="same", kernel_initializer=self.weights_init,
                                        strides=self.params_json["gen_conv_strides"][layer_n])(conv_layer)
            
            else:
                conv_layer = Conv2DTranspose(filters=self.params_json["gen_conv_filters"][layer_n], 
                                        kernel_size=self.params_json["gen_conv_kernel_size"][layer_n],
                                        padding="same", strides=self.params_json["gen_conv_strides"][layer_n])(conv_layer)

            conv_layer = Activation(self.params_json["gen_conv_activations"][layer_n])(conv_layer)
            conv_layer = Dropout(rate=self.params_json["gen_conv_dropout_rates"][layer_n])(conv_layer)
            conv_layer = BatchNormalization()(conv_layer)
        
        output_layer = Conv2D(filters=self.params_json["input_shape"][-1], kernel_size=3, padding="same")(conv_layer)
        output_layer = Activation(self.params_json["gen_out_activations"])(output_layer)
        self.generator = Model(input_layer, output_layer)

    
    def _build_model_(self):

        self.discriminator.compile(
            optimizer=RMSprop(learning_rate=self.params_json["dis_learning_rate"]),
            loss=self.params_json["dis_loss_function"],
            metrics=self.params_json["dis_metrics"]
        )
        self.set_trainable(model=self.discriminator, value=False)

        model_input = Input(shape=(self.params_json["noise_dim"], ))
        model_output = self.discriminator(self.generator(model_input))
        
        self.model = Model(model_input, model_output)
        self.model.compile(
            optimizer=Adam(learning_rate=self.params_json["entire_model_learning_rate"]),
            loss=self.params_json["entire_model_loss_function"],
            metrics=self.params_json["entire_model_metrics"]
        )

        self.set_trainable(model=self.discriminator, value=True)

    def _train_dis_(self, train_tensor, batch_size):

        random_idx = np.random.randint(0, train_tensor.shape[0] - 1, batch_size)
        valid_labels = np.ones(batch_size)
        fake_labels = np.zeros(batch_size)
        
        noise = np.random.normal(0.98, 1.29, (batch_size, self.params_json["noise_dim"]))
        true_samples = train_tensor[random_idx]
        gen_samples = self.generator.predict(noise)

        real_history = self.discriminator.train_on_batch(true_samples, valid_labels)
        fake_history = self.discriminator.train_on_batch(gen_samples, fake_labels)

        return [real_history, fake_history]

    def _train_gen_(self, train_tensor, batch_size):

        noise = np.random.normal(0.98, 1.29, (batch_size, self.params_json["noise_dim"]))
        valid_labels = np.ones(batch_size)
        return self.model.train_on_batch(noise, valid_labels)

    def save_samples(self, samples_number, gen_folder):
        
        plt.style.use("dark_background")
        curent_epoch_samples = os.path.join(gen_folder, f"generated_samples_{self.epoch_iterator}.png")

        samples_number_sq = int(np.sqrt(samples_number))
        fig, axis = plt.subplots()

        if not (self.params_json["input_shape"][-1] == 1):
            show_tensor = np.zeros((samples_number_sq * self.params_json["input_shape"][0], samples_number_sq * self.params_json["input_shape"][1], self.params_json["input_shape"][-1]))
        
        else:
            show_tensor = np.zeros((samples_number_sq * self.params_json["input_shape"][0], samples_number_sq * self.params_json["input_shape"][1]))

        noise = np.random.normal(0.98, 1.29, (samples_number, self.params_json["noise_dim"]))
        gen_samples = self.generator.predict(noise)

        sample_number = 0

        for i in range(samples_number_sq):
            for j in range(samples_number_sq):
                
                if not (self.params_json["input_shape"][-1] == 1):
                    show_tensor[i * self.params_json["input_shape"][0]: (i + 1) * self.params_json["input_shape"][0],
                                j * self.params_json["input_shape"][0]: (j + 1) * self.params_json["input_shape"][1], :] = gen_samples[sample_number]
                
                else:
                    show_tensor[i * self.params_json["input_shape"][0]: (i + 1) * self.params_json["input_shape"][0],
                                j * self.params_json["input_shape"][0]: (j + 1) * self.params_json["input_shape"][1]] = gen_samples[sample_number]

                sample_number += 1
        
        
        axis.imshow(show_tensor, cmap="inferno")
        fig.savefig(curent_epoch_samples)

    def train_model(self, epochs, batch_size, run_folder, epoch_per_save, train_tensor):

        gs_folder = os.path.join(run_folder, "gen_samples_folder")
        weights_folder = os.path.join(run_folder, "model_weights_folder")
        
        folders = [gs_folder, weights_folder]
        for folder in folders:
            
            if not os.path.exists(folder):
                os.mkdir(folder)
        
        for epoch in range(epochs):

            self.dis_losses.append(self._train_dis_(train_tensor=train_tensor, batch_size=batch_size))
            self.gen_losses.append(self._train_gen_(train_tensor=train_tensor, batch_size=batch_size))

            if epoch % epoch_per_save == 0:
                self.save_samples(gen_folder=gs_folder, samples_number=25)

            self.epoch_iterator += 1

        generator_weights_path = os.path.join(weights_folder, "generator_model_weights.weights.h5")
        discriminator_weights_path = os.path.join(weights_folder, "discriminator_model_weights.weights.h5")
        entire_model_weights_path = os.path.join(weights_folder, "entire_model_weights.weights.h5")

        self.generator.save_weights(filepath=generator_weights_path)
        self.discriminator.save_weights(filepath=discriminator_weights_path)
        self.model.save_weights(filepath=entire_model_weights_path)
    
    def load_weights(self, filepath):

        for ws_file in os.listdir(filepath):
            
            ws_path = os.path.join(filepath, ws_path)
            if "generator" in ws_file:
                self.generator.load_weights(ws_path)
            
            elif "discriminator" in ws_file:
                self.discriminator.load_weights(ws_path)
            
            elif "entire_model" in ws_file:
                self.model.load_weights(ws_path)








class VarEncoder:

    def __init__(self, params_json=None, filepath=None) -> None:
        
        self.params_json = params_json

        if self.params_json is None:

            if filepath is None:
                filepath = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\VarAutoEncoder.json"
                
            self.load_params(filepath=filepath)
        

        self.encoder_conv_layers_n = len(self.params_json["encoder_conv_filters"])
        self.encoder_dense_layers_n = len(self.params_json["encoder_dense_units"])
        self.decoder_layers_n = len(self.params_json["decoder_conv_filters"])
        self.epoch_iterator = 0
        self.weights_init = RandomNormal(mean=self.params_json["weights_init"]["mean"], stddev=self.params_json["weights_init"]["stddev"])
        self.model_loss = []
    
        self._build_encoder_()
        self._build_decoder_()
        self._build_model_()
        self._load_hiden_dim_()

        self.save_params(filepath=filepath)
        
    
    def _load_hiden_dim_(self, json_log=None):
        
        if json_log is None:
            json_log = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_save\\AE_save\\hiden_dim_storage.json"

        with open(json_log, "r") as json_file:
            self.hiden_dim = js.load(json_file)
        
        for class_label in self.hiden_dim.keys():
            self.hiden_dim[class_label] = np.asarray(self.hiden_dim[class_label])
    
    def _build_encoder_(self):
        
        def sampling(args):

            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon


        input_layer = Input(shape=self.params_json["input_shape"])
        encoder_layer = input_layer

        for layer_number in range(self.encoder_conv_layers_n):

            encoder_layer = Conv2D(filters=self.params_json["encoder_conv_filters"][layer_number], kernel_size=self.params_json["encoder_conv_kernel_size"][layer_number],
                                   padding="same", strides=self.params_json["encoder_conv_strides"][layer_number],
                                   kernel_initializer=self.weights_init)(encoder_layer)
            
            encoder_layer = Activation(self.params_json["encoder_conv_activations"][layer_number])(encoder_layer)
            encoder_layer = Dropout(rate=self.params_json["encoder_conv_dropout"][layer_number])(encoder_layer)
            encoder_layer = BatchNormalization()(encoder_layer)
        
        self.saved_shape = encoder_layer.shape[1:]
        dense_layer = Flatten()(encoder_layer)

        for layer_number in range(self.encoder_dense_layers_n):

            dense_layer = Dense(units=self.params_json["encoder_dense_units"][layer_number], activation=self.params_json["encoder_dense_activations"][layer_number])(dense_layer)
            dense_layer = Dropout(rate=self.params_json["encoder_dense_dropout_rates"][layer_number])(dense_layer)
        
        mean_layer = Dense(units=self.params_json["hiden_dim"], activation=self.params_json["encoder_out_activations"])(dense_layer)
        std_layer = Dense(units=self.params_json["hiden_dim"], activation=self.params_json["encoder_out_activations"])(dense_layer)
        output_layer = Lambda(sampling)([mean_layer, std_layer])

        self.encoder = Model(input_layer, output_layer)
    
    def _build_decoder_(self):

        input_layer = Input(shape=(self.params_json["hiden_dim"], ))
        rec_layer = Dense(units=np.prod(self.saved_shape))(input_layer)
        rec_layer = Reshape(target_shape=self.saved_shape)(rec_layer)
        
        decoder_layer = rec_layer
        for layer_number in range(self.decoder_layers_n):

            decoder_layer = Conv2DTranspose(filters=self.params_json["decoder_conv_filters"][layer_number], kernel_size=self.params_json["decoder_conv_kernel_size"][layer_number],
                                            padding="same", strides=self.params_json["decoder_conv_strides"][layer_number],
                                            kernel_initializer=self.weights_init)(decoder_layer)
            
            decoder_layer = Activation(self.params_json["decoder_conv_activations"][layer_number])(decoder_layer)
            decoder_layer = Dropout(rate=self.params_json["decoder_conv_dropout"][layer_number])(decoder_layer)
            decoder_layer = BatchNormalization()(decoder_layer)
        
        output_layer = Conv2D(filters=self.params_json["input_shape"][-1], strides=1, padding="same", kernel_size=3)(decoder_layer)
        output_layer = Activation(self.params_json["decoder_out_activations"])(output_layer)
        self.decoder = Model(input_layer, output_layer)
    
    def _build_model_(self):

        model_input_layer = Input(shape=self.params_json["input_shape"])
        model_output_layer = self.decoder(self.encoder(model_input_layer))
        self.model = Model(model_input_layer, model_output_layer)
        self.model.compile(loss="mse", metrics=["mae"], optimizer=RMSprop(learning_rate=0.01))
    
    def train(self, run_folder, train_tensor, train_labels, epochs, batch_size, epoch_per_save):

        
        run_folder = run_folder
        if not os.path.exists(run_folder):
            os.mkdir(run_folder)
        entire_model_weights_folder = os.path.join(run_folder, "entire_model_weights.weights.h5")

        for epoch in range(epochs):
            
            random_idx = np.random.randint(0, train_tensor.shape[0] - 1, batch_size)
            train_batch = train_tensor[random_idx]

            self.model_loss.append(self.model.train_on_batch(train_batch, train_batch))
            if epoch % epoch_per_save == 0:
                self.save_samples(samples_number=100, data_tensor=train_tensor, run_folder=run_folder)
            
            self.epoch_iterator += 1
        
        hiden_dim_json = os.path.join(run_folder, "hiden_dim_storage.json")
        self.generate_encoded_dim(data_tensor=train_tensor, general_labels=train_labels, hiden_dim_json=hiden_dim_json)
        self.model.save_weights(filepath=entire_model_weights_folder)

    
    def generate_encoded_dim(self, general_labels, data_tensor, hiden_dim_json):
        
        encoded_vectors = self.encoder.predict(data_tensor)
        self.hiden_dim = {class_name: [] for class_name in self.params_json["classes_dis"].values()}
        
        for (encoded_point, class_label) in zip(encoded_vectors, general_labels):

            class_name = self.params_json["classes_dis"][class_label]
            self.hiden_dim[class_name].append(encoded_point.tolist())
        
        with open(hiden_dim_json, "w") as json_file:
            js.dump(self.hiden_dim, json_file)
        

    def load_weights(self, filepath=None):
        
        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_save\\AE_save\\entire_model_weights.weights.h5"
        self.model.load_weights(filepath)
    
    def save_params(self, filepath=None):

        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\VarAutoEncoder.json"

        with open(filepath, "w") as file:
            js.dump(self.params_json, file)

    def load_params(self, filepath):

        with open(filepath, "r") as file:
            self.params_json = js.load(file)


    def save_samples(self, samples_number, data_tensor, run_folder):
        
        gen_samples_folder = os.path.join(run_folder, "generated_samples")
        if not os.path.exists(gen_samples_folder):
            os.mkdir(gen_samples_folder)
        curent_epoch_samples = os.path.join(gen_samples_folder, f"generated_samples_{self.epoch_iterator}.png")

        samples_number_sq = int(np.sqrt(samples_number))
        fig, axis = plt.subplots()

        if not (self.params_json["input_shape"][-1] == 1):
            show_tensor = np.zeros((samples_number_sq * self.params_json["input_shape"][0], samples_number_sq * self.params_json["input_shape"][1], self.params_json["input_shape"][-1]))
        
        else:
            show_tensor = np.zeros((samples_number_sq * self.params_json["input_shape"][0], samples_number_sq * self.params_json["input_shape"][1]))

        random_idx = np.random.randint(0, data_tensor.shape[0] - 1, samples_number)
        encoded_vectors = self.encoder.predict(data_tensor[random_idx])
        decoded_images = self.decoder.predict(encoded_vectors)
        sample_number = 0

        for i in range(samples_number_sq):
            for j in range(samples_number_sq):
                
                if not (self.params_json["input_shape"][-1] == 1):
                    show_tensor[i * self.params_json["input_shape"][0]: (i + 1) * self.params_json["input_shape"][0],
                                j * self.params_json["input_shape"][0]: (j + 1) * self.params_json["input_shape"][1], :] = decoded_images[sample_number]
                
                else:
                    show_tensor[i * self.params_json["input_shape"][0]: (i + 1) * self.params_json["input_shape"][0],
                                j * self.params_json["input_shape"][0]: (j + 1) * self.params_json["input_shape"][1]] = decoded_images[sample_number]

                sample_number += 1
        
        
        axis.imshow(show_tensor, cmap="inferno")
        fig.savefig(curent_epoch_samples)







class RnnConv:

    def __init__(self, params_json=None, params_path=None) -> None:
        
        self.params_json = params_json
        if self.params_json is None:
            self._load_params_(filepath=params_path)

        self.encoder_layers_n = len(self.params_json["encoder_params"]["filters"])
        self.decoder_layers_n = len(self.params_json["decoder_params"]["units"])

        self._build_encoder_()
        self._build_decoder_()
        self._build_model_()
    
    def _load_params_(self, filepath):
        
        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\RnnConv.json"

        with open(filepath, "r") as json_file:
            self.params_json = js.load(json_file)
    
    def _save_params(self, filepath):

        with open(filepath, "w") as json_file:
            js.dump(json_file, self.params_json)

    def _build_encoder_(self):
        
        encoder_params = self.params_json["encoder_params"]
        input_layer = Input(shape=self.params_json["input_shape"])
        conv_layer = input_layer

        for layer_n in range(self.encoder_layers_n):

            conv_layer = Conv2D(filters=encoder_params["filters"][layer_n],
                                kernel_size=encoder_params["kernel_size"][layer_n],
                                strides=encoder_params["strides"][layer_n],
                                padding="same")(conv_layer)
            
            conv_layer = Activation(encoder_params["activations"][layer_n])(conv_layer)
            if not encoder_params["single_dropout"]:
                conv_layer = Dropout(encoder_params["dropout_rates"][layer_n])(conv_layer)
            
            conv_layer = BatchNormalization()(conv_layer)
        
        conv_layer = Conv2D(filters=1, kernel_size=3, strides=1)(conv_layer)
        conv_layer = Activation(encoder_params["output_activation"])(conv_layer)

        output_layer = ConvSequence(units=self.params_json["decoder_params"]["units"][0], sequence_l=None, samples_n=100)(conv_layer)
        self.encoder = Model(inputs=input_layer, outputs=output_layer)

    def _build_decoder_(self):
        
        decoder_params = self.params_json["decoder_params"]

        input_layer = Input(shape=(None, self.params_json["decoder_params"]["units"][0]))
        sequence_input_layer = Input(shape=(None, ))
        embedding_layer = Embedding(input_dim=self.params_json["total_labels_n"], output_dim=decoder_params["embedding_dim"])(sequence_input_layer)

        for layer_n in range(self.decoder_layers_n):
            
            if layer_n == (self.decoder_layers_n - 1):
                break
            
            if layer_n == 0:

                if decoder_params["bidirectional"][layer_n]:
                    lstm_layer = Bidirectional(LSTM(units=decoder_params["units"][layer_n], return_sequences=True))(embedding_layer, initial_state=input_layer)

                else:
                    lstm_layer = LSTM(units=decoder_params["units"][layer_n], return_sequence=True)(embedding_layer, initial_state=input_layer)
            
            else:

                if decoder_params["bidirectional"][layer_n]:
                    lstm_layer = Bidirectional(LSTM(units=decoder_params["units"][layer_n], return_sequences=True))(lstm_layer)

                else:
                    lstm_layer = LSTM(units=decoder_params["units"][layer_n], return_sequence=True)(lstm_layer)

            if not decoder_params["single_dropout"]:
                lstm_layer = Dropout(rate=decoder_params["dropout_rates"][layer_n])(lstm_layer)
        
        lstm_layer = LSTM(units=decoder_params["units"][-1])(lstm_layer)
        if decoder_params["single_dropout"]:
            lstm_layer = Dropout(rate=0.26)(lstm_layer)
        
        decoder_output_layer = Dense(units=self.params_json["total_labels_n"], activation="softmax")(lstm_layer)
        self.decoder = Model(inputs=[input_layer, sequence_input_layer], outputs=decoder_output_layer)

    def _build_model_(self):
        
        model_input_layer = Input(shape=self.params_json["input_shape"])
        sequence_layer = Input(shape=(None, ))
        model_output_layer = self.decoder([self.encoder(model_input_layer), sequence_layer])

        self.model = Model(inputs=model_input_layer, outputs=model_output_layer)
        self.model.compile(loss=CategoricalCrossentropy(), optimizer="rmsprop")