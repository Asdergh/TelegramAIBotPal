import numpy as np
import random as rd
import matplotlib.pyplot as plt
import json as js
import os

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LayerNormalization, Layer, Dropout, Multiply
from tensorflow.keras.layers import Dense, Flatten, UpSampling2D, Activation, Add, Concatenate, Normalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model


class DownConvLayer(Layer):

    def __init__(self, filters, init, kernel_size=4, activation="tanh", strides=2, dropout_rate=0.76):

        conv_layer = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", strides=strides, kernel_initializer=init)
        norm_layer = Normalization(axis=-1, mean=None, variance=None, invert=False)
        dropout_layer = Dropout(rate=dropout_rate)
        activation_layer = Activation(activation=activation)

        self.layers = [conv_layer, activation_layer, norm_layer, dropout_layer]
    
    def __call__(self, input_layer):
        
        x = input_layer
        for layer in self.layers:
            x = layer(x)
        
        return x
    
class UpConvLayer(Layer):

    def __init__(self, filters, init, kernel_size=4, activation="tanh", strides=1, dropout_rate=0.76):

        up_layer = UpSampling2D(size=2)
        conv_layer = Conv2D(filters=filters, padding="same", kernel_size=kernel_size, strides=strides, kernel_initializer=init)
        norm_layer = Normalization(axis=-1, mean=None, variance=None, invert=False)
        dropout_layer = Dropout(rate=dropout_rate)
        activation_layer = Activation(activation=activation)
        mul_layer = Multiply()

        self.layers = [up_layer, conv_layer, activation_layer, norm_layer, dropout_layer, mul_layer]
    
    def __call__(self, inputs):
        

        x = inputs[0]
        for (layer_n, layer) in enumerate(self.layers):
            
            if layer_n == (len(self.layers) - 1):
                break
            x = layer(x)
        
        x = self.layers[-1]([x, inputs[-1]])
        return x


        
class CycleGAN():

    def __init__(self, params_json=None, params_path=None):

        self.params_json = params_json
        if self.params_json is None:
            self.__load_params__(filepath=params_path)
        self.__save_params__()

        self.input_shape = self.params_json["input_shape"]
        self.dis_prob_patch = (int(self.input_shape[0] // (2 ** self.params_json["discriminator_params"]["layers_n"])),  
                              int(self.input_shape[1] // (2 ** self.params_json["discriminator_params"]["layers_n"])), 1)
        self.weights_init = RandomNormal(mean=2.1, stddev=3.98)
        self.dis_learning_history = []
        self.gen_learning_history = []

        self.__build_model__()    
    
    def __load_params__(self, filepath=None):

        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\CycleGAN.json"

        with open(filepath, "r") as json_file:
            self.params_json = js.load(json_file)
    
    def __save_params__(self, filepath=None):

        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\CycleGAN.json"

        with open(filepath, "w") as json_file:
            js.dump(self.params_json, json_file)
        
    
    def __build_discriminator__(self):

        dis_params = self.params_json["discriminator_params"]
        filters_n = dis_params["filters_n"]
        layers_n = dis_params["layers_n"]
        last_activation = dis_params["last_activation"]

        input_layer = Input(shape=self.input_shape)
        conv_layer = DownConvLayer(filters=filters_n, activation="tanh", init=self.weights_init)(input_layer)
        for layer_n in range(layers_n):
            
            if layer_n == (layers_n - 1):
                conv_layer = DownConvLayer(filters=filters_n, activation="tanh", init=self.weights_init, strides=1)(conv_layer)
            
            else:
                conv_layer = DownConvLayer(filters=filters_n, activation="tanh", init=self.weights_init)(conv_layer)

            filters_n *= 2
        
        output_layer = Conv2D(filters=1, kernel_size=4, strides=1, padding="same")(conv_layer)
        output_layer = Activation(last_activation)(output_layer)

        discriminator = Model(inputs=input_layer, outputs=output_layer)
        return discriminator
        

    def __build_generator_unet__(self):
        
        filters = self.params_json["generator_params"]["filters_n"]
        down_activation = self.params_json["generator_params"]["down_activation"]
        up_activation = self.params_json["generator_params"]["down_activation"]
        last_activation = self.params_json["generator_params"]["last_activation"]
    
        input_layer = Input(shape=self.input_shape)
        down_layer_0 = DownConvLayer(filters=filters, init=self.weights_init, activation=down_activation)(input_layer)
        down_layer_1 = DownConvLayer(filters=filters * 2, init=self.weights_init, activation=down_activation)(down_layer_0)
        down_layer_2 = DownConvLayer(filters=filters * 4, init=self.weights_init, activation=down_activation)(down_layer_1)
        down_layer_3 = DownConvLayer(filters=filters * 8, init=self.weights_init, activation=down_activation)(down_layer_2)

        up_layer_0 = UpConvLayer(filters=filters * 4, init=self.weights_init, activation=up_activation)([down_layer_3, down_layer_2])
        up_layer_1 = UpConvLayer(filters=filters * 2, init=self.weights_init, activation=up_activation)([up_layer_0, down_layer_1])
        up_layer_2 = UpConvLayer(filters=filters, init=self.weights_init, activation=up_activation)([up_layer_1, down_layer_0])

        up_layer_3 = UpSampling2D(size=2)(up_layer_2)
        up_layer_3 = Conv2D(filters=self.input_shape[-1], kernel_size=4, padding="same", strides=1, kernel_initializer=self.weights_init)(up_layer_3)
        output_layer = Activation(last_activation)(up_layer_3)

        generator = Model(inputs=input_layer, outputs=output_layer)
        return generator
        



    def __build_model__(self):

        self.discriminator_A = self.__build_discriminator__()
        self.discriminator_B = self.__build_discriminator__()

        self.discriminator_A.compile(loss="mse", optimizer="adam")
        self.discriminator_B.compile(loss="mse", optimizer="adam")

        self.generator_A = self.__build_generator_unet__()
        self.generator_B = self.__build_generator_unet__()
        
        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False

        input_image_A = Input(shape=self.input_shape)
        input_image_B = Input(shape=self.input_shape)
        
        
        fake_A = self.generator_A(input_image_B)
        fake_B = self.generator_B(input_image_A)

        dis_A = self.discriminator_A(fake_A)
        dis_B = self.discriminator_B(fake_B)

        rec_A = self.generator_B(fake_A)
        rec_B = self.generator_A(fake_B)
        
        self.model = Model(inputs=[input_image_A, input_image_B], outputs=[dis_A, dis_B,
                                                                           fake_A, fake_B, 
                                                                           rec_A, rec_B])
        self.model.compile(loss=["binary_crossentropy", "binary_crossentropy", 
                                 "mse", "mse", 
                                 "mse", "mse"], optimizer="adam")
        
        self.discriminator_A.trainable = True
        self.discriminator_B.trainable = True
    
    def __train_discriminator__(self, images_A, images_B, batch_size):

        valid = np.ones((batch_size, ) + self.dis_prob_patch)
        fake = np.zeros((batch_size, ) + self.dis_prob_patch)

        fake_A = self.generator_A.predict(images_B)
        fake_B = self.generator_B.predict(images_A)

        real_loss_A = self.discriminator_A.train_on_batch(images_A, valid)
        fake_loss_A = self.discriminator_A.train_on_batch(fake_A, fake)
        A_losses = [real_loss_A, fake_loss_A]
        
        real_loss_B = self.discriminator_A.train_on_batch(images_B, valid)
        fake_loss_B = self.discriminator_B.train_on_batch(fake_B, fake)        
        B_loss = [real_loss_B, fake_loss_B]


        A_avg_loss = 0.5 * sum(A_losses)
        B_avg_loss = 0.5 * sum(B_loss)

        return [A_avg_loss, real_loss_A, fake_loss_A,
                B_avg_loss, real_loss_B, fake_loss_B]
    
    def __train_generator__(self, images_A, images_B, batch_size):

        valid = np.ones((batch_size, ) + self.dis_prob_patch)
        return self.model.train_on_batch([images_A, images_B], [valid, valid,
                                                                images_A, images_B,
                                                                images_B, images_B])
    
    def __generate_samples__(self, epoch, images_A, images_B, run_folder=None, samples_number=6):

        generation_folder = os.path.join(run_folder, "generated_train_samples")
        if not os.path.exists(generation_folder):
            os.mkdir(generation_folder)
        

        tensor_shape = (4 * self.input_shape[0], samples_number * self.input_shape[1], self.input_shape[2])
        show_tensor = np.zeros(shape=tensor_shape)

        random_idx = np.random.randint(0, min(images_A.shape[0], images_B.shape[0]), samples_number)
        batch_A = images_A[random_idx]
        batch_B = images_B[random_idx]

        gen_fake_A = self.generator_A.predict(batch_A)
        gen_fake_B = self.generator_B.predict(batch_B)
        
        rec_A = self.generator_A.predict(gen_fake_B)
        rec_B = self.generator_B.predict(gen_fake_A)
        
        gen_samples = [gen_fake_A, gen_fake_B, rec_A, rec_B]
        for (samples_n, samples) in enumerate(gen_samples):
            for (sample_n, sample) in enumerate(samples):
                
                if self.input_shape[-1] == 1:

                    sample = np.resize(sample, (self.input_shape[0], self.input_shape[1]))
                    show_tensor[samples_n * self.input_shape[0]: (samples_n + 1) * self.input_shape[0],
                                sample_n * self.input_shape[1]: (sample_n + 1) * self.input_shape[1]] = sample
                
                else:
                    show_tensor[samples_n * self.input_shape[0]: (samples_n + 1) * self.input_shape[0],
                            sample_n * self.input_shape[1]: (sample_n + 1) * self.input_shape[1], :] = sample

        plt.style.use("dark_background")
        fig, axis = plt.subplots()
        axis.imshow(show_tensor, cmap="inferno")

        gen_samples_path = os.path.join(generation_folder, f"gen_sample_number_{epoch}.png")
        fig.savefig(gen_samples_path)
    

                
    def train_model(self, images_A, images_B, batch_size, epochs, epochs_per_save=10):

        run_folder = self.params_json["run_folder"]
        for epoch in range(epochs):

            random_idx = np.random.randint(0, min(images_A.shape[0], images_B.shape[1]), batch_size)
            batch_A = images_A[random_idx]
            batch_B = images_B[random_idx]
            
            dis_train_res = self.__train_discriminator__(images_A=batch_A, images_B=batch_B, batch_size=batch_size)
            gen_train_res = self.__train_generator__(images_A=batch_A, images_B=batch_B, batch_size=batch_size)
            
            if (epoch % epochs_per_save == 0):
                self.__generate_samples__(epoch=epoch, images_A=batch_A, images_B=batch_B, run_folder=run_folder)
            
            self.dis_learning_history.append(dis_train_res)
            self.gen_learning_history.append(gen_train_res)
        
        weights_path = os.path.join(run_folder, "weights.weights.h5")
        self.model.save_weights(filepath=weights_path)

    
    


    