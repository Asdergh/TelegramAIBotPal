import numpy as np
import random as rd
import matplotlib.pyplot as plt
import librosa as lb
import soundfile as sf
import cv2
import os
import time as t
import tensorflow.keras.backend as K

from tensorflow.data import Dataset
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

target_size = (450, 450, 3)
batch_size = 25

class ImageGenerator:

    def __init__(self, data_path, data_type="train", target_size=(120, 128), categorical=False, batch_size=None) -> None:
        
        self.data_path = data_path
        self.data_type = data_type
        self.target_size = target_size
        self.categorical = categorical
        self.batch_size = batch_size

        self._curent_folder_ = None
        for data_folder in os.listdir(self.data_path):

            if data_folder.lower() == self.data_type:
                self._curent_folder_ = os.path.join(self.data_path, data_folder)

        if self._curent_folder_ is None:
            raise ValueError("data_type variable must have on the following values: [train, test, validation], to empasize data useabbility!!!")
        
        self._all_classes_ = os.listdir(self._curent_folder_)

    def __iter__(self):
        
        while True:
            yield self.__collect_data__()
    
    def __next__(self):

        return self.__collect_data__()
    

    def __one_sample__(self):

        random_idx = np.random.randint(0, len(self._all_classes_), dtype="int")
        sub_folder = os.path.join(self._curent_folder_, self._all_classes_[random_idx])
        sample_path = os.path.join(sub_folder, rd.choice(os.listdir(sub_folder)))

        sample = cv2.imread(sample_path)
        sample = cv2.resize(sample, self.target_size)
        return (random_idx, sample)

    def __batch_sample__(self):
        
        random_idx = np.random.randint(0, len(self._all_classes_), self.batch_size, dtype="int")
        sub_folders = [os.path.join(self._curent_folder_, self._all_classes_[idx]) for idx in random_idx]
        sample = []

        for sub_folder in sub_folders:

            pathes = os.listdir(sub_folder)
            sample_path = os.path.join(sub_folder, rd.choice(pathes))
            sub_sample = cv2.imread(sample_path)
            sub_sample = cv2.resize(sub_sample, self.target_size)

            sample.append(sub_sample)

        sample = np.asarray(sample)
        return (random_idx, sample)
    
    def __one_label__(self, idx):

        label = np.zeros(len(self._all_classes_))
        label[idx] = 1.0
        return label
    
    def __batch_label__(self, idx):

        labels = [np.zeros(len(self._all_classes_)) for _ in range(batch_size)]
        for label_n, (idx, label) in enumerate(zip(idx, labels)):
                        
            label[idx] = 1.0
            labels[label_n] = label
        
        return label

    def __collect_data__(self):

        if self.batch_size == 1 or self.batch_size is None:
            random_idx, sample = self.__one_sample__()
            
        else:
            random_idx, sample = self.__batch_sample__()
                
        if self.categorical:
                
            if self.batch_size == 1 or self.batch_size is None:
                label = self.__one_label__(idx=random_idx)
                
            else:
                label = self.__batch_label__(idx=random_idx)

            return (sample, label)
            
        else:
            return sample
        
    
            
    

   

            


time_s = t.time()
image_generator = ImageGenerator(data_path="c:\\Users\\1\\Desktop\\data_files\\human_emotions", target_size=target_size[:-1], categorical=False, batch_size=1000)
data_tensor = next(image_generator)
time_e = t.time()
print(data_tensor.shape, time_e - time_s)
# dataset = Dataset
# dataset = dataset.from_generator(generator=image_generator, output_types="float")
# dataset_batched = dataset.batch(batch_size=batch_size)


    
        

# input_layer = Input(shape=target_size)
# conv_layer = input_layer
# for _ in range(3):
      
#     conv_layer = Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(conv_layer)
#     conv_layer = Activation("tanh")(conv_layer)

# conv_layer = Flatten()(conv_layer)
# output_layer = Dense(units=10, activation="softmax")(conv_layer)
# model = Model(inputs=input_layer, outputs=output_layer)
# model.compile(optimizer="adam", loss="categorical_crossentropy")





