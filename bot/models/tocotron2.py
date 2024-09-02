import numpy as np
import random as rd
import matplotlib.pyplot as plt
import json as js
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import AbsoluteError
from tensorflow.keras import Model

class Tocotron:

    def __init__(self, params_json=None, params_path=None) -> None:
        
        self.params_json = params_json
        if self.params_json is None:
            
            if params_path is None:
                params_path = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\models_params\\Tacotron2.json"
            self._load_params_(filepath=params_json)
        
    
    def _load_params_(self, filepath=None):

        with open(filepath, "r") as json_file:
            self.params_json = js.load(json_file)
    
    def _save_params_(self, filepath):

        with open(filepath, "w") as json_file:
            js.dump(self.params_json, json_file)
    
    def _build_pernet_(self):
        pass
    
        