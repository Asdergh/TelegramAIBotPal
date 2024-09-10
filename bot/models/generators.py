import numpy as np
import random as rd
import librosa as lb
import os
import cv2
import time as t
import soundfile as sf
import librosa as ls
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from librosa.feature import melspectrogram as mel_s
from tensorflow.data import Dataset
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical



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

        labels = [np.zeros(len(self._all_classes_)) for _ in range(self.batch_size)]
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
    

class TextMelGenerator:

    def __init__(self, data_path, batch_size=None, max_mel_len=480) -> None:
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_mel_len = max_mel_len

    def __iter__(self):

        while True:

            if self.batch_size == 1 or self.batch_size is None:
                sample = self.__one_sample__()

            else:
                sample = self.__batch_sample__()
            
            yield sample
    
    def __next__(self):

        if self.batch_size == 1 or self.batch_size is None:
            sample = self.__one_sample__()

        else:
            sample = self.__batch_sample__()

        return sample
        
    def __collect_data__(self, folder, files_list):
        
        if files_list:

            for file in files_list:

                curent_file = os.path.join(folder, file)
                if "txt" in file:
                
                    with open(curent_file, "r") as file:
                        text = file.read()


                elif "wav" in file:

                    wave, sr = ls.load(curent_file)
                    mel = mel_s(y=wave, sr=sr)
                    
                    if mel.shape[1] > self.max_mel_len:
                        mel = mel[:, :self.max_mel_len]
                    
                    else:
                        
                        tmp_mel = np.zeros((mel.shape[0], self.max_mel_len))
                        value_index = 0
                        mel_value_index = 0
                        while value_index < tmp_mel.shape[1]:
                            
                            if mel_value_index == mel.shape[1]:
                                mel_value_index = 0

                            tmp_mel[:, value_index] = mel[:, mel_value_index]
                            value_index += 1
                            mel_value_index += 1
                        
                        mel = tmp_mel
        
        else:

            text = ""
            mel = np.zeros((128, self.max_mel_len))
        
        return (text, mel)

    def __one_sample__(self):

        random_folder = os.path.join(self.data_path, rd.choice(os.listdir(self.data_path)))
        random_sub_folder = os.path.join(random_folder, rd.choice(os.listdir(random_folder)))
        random_file = rd.choice(os.listdir(random_sub_folder))

        same_files = [file_name for file_name in os.listdir(random_sub_folder) 
                      if file_name[:file_name.find(".")] == random_file[:random_file.find(".")] 
                      and (file_name[file_name.find(".") + 1] == "o" or "wav" in file_name)]
        if same_files:
            text, mel = self.__collect_data__(folder=random_sub_folder, files_list=same_files)
        
        else:
            random_file = rd.choice(os.listdir(random_sub_folder))
            same_files = [file_name for file_name in os.listdir(random_sub_folder) 
                        if file_name[:file_name.find(".")] == random_file[:random_file.find(".")] 
                        and (file_name[file_name.find(".") + 1] == "o" or "wav" in file_name)]
            
            text, mel = self.__collect_data__(folder=random_sub_folder, files_list=same_files)
            
        sample = {
            "text": text,
            "mel_sp": mel
        }
        
        return sample
    
    def __batch_sample__(self):

        sample = {
            "text": [],
            "mel_sp": []
        }
        for sample_n in range(self.batch_size):

            random_folder = os.path.join(self.data_path, rd.choice(os.listdir(self.data_path)))
            random_sub_folder = os.path.join(random_folder, rd.choice(os.listdir(random_folder)))
            random_file = rd.choice(os.listdir(random_sub_folder))
        
            same_files = [file_name for file_name in os.listdir(random_sub_folder) 
                        if file_name[:file_name.find(".")] == random_file[:random_file.find(".")] 
                        and (file_name[file_name.find(".") + 1] == "o" or "wav" in file_name)]
            
            if same_files:
                text, mel = self.__collect_data__(folder=random_sub_folder, files_list=same_files)
                sample["text"].append(text)
                sample["mel_sp"].append(mel)
            
            else:

                random_file = rd.choice(os.listdir(random_sub_folder))
                same_files = [file_name for file_name in os.listdir(random_sub_folder) 
                            if file_name[:file_name.find(".")] == random_file[:random_file.find(".")] 
                            and (file_name[file_name.find(".") + 1] == "o" or "wav" in file_name)]
                
                text, mel = self.__collect_data__(folder=random_sub_folder, files_list=same_files)
                sample["text"].append(text)
                sample["mel_sp"].append(mel)
                
        sample["mel_sp"] = np.asarray(sample["mel_sp"], dtype="float")
        return sample



    

  

    
    
   