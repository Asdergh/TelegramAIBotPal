import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import json as js
import tensorflow.keras.backend as K

from PIL import Image
from matplotlib.animation import FuncAnimation
from conv_var_ae import VarEncoder
from rnn import RNN
from rnn_ae import RNN_AE
from matplotlib.animation import FuncAnimation
from telegram.ext import Application, filters, ContextTypes
from telegram.ext import MessageHandler, CommandHandler
from telegram import Update

class BotCore:

    def __init__(self) -> None:
        

        
        self.seq2em_model = RNN(filepath="C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\RNN.json")
        self.em2img_model = VarEncoder(filepath="C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\VarAutoEncoder.json")
        self.rnn_ae_model = RNN_AE()

        self.seq2em_model.load_weights()
        self.em2img_model.load_weights()
        self.rnn_ae_model.load_weights()

        self.saved_images_folder = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\save_images_folder"
        self.saved_voices_folder = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\save_images_folder"
        self.app_builder = Application.builder()
        self.app_builder.token("7405563731:AAHhDPtAHWXzi-ZCRh-F-uWMBtBZSyrv3gg")
        self.application = self.app_builder.build()

        self.message_handler = MessageHandler(filters=filters.TEXT, callback=self.__msg_answer__)
        self.image_handler = MessageHandler(filters=filters.CAPTION, callback=self.__image_generation__)
                
        all_handlers = [self.message_handler, self.image_handler]
        for handler in all_handlers:
            self.application.add_handler(handler)
        self.application.run_polling(poll_interval=3)
        
        
    
    
    async def __image_generation__(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pass

    async def __msg_answer__(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        msg = update.message.text
        print(msg)
        msg_reply = self.__handle_text_msg__(msg=msg)
        print(msg_reply)
        await update.message.reply_text(msg_reply)


    def __handle_text_msg__(self, msg):

        self.rnn_ae_model.expand_tokenizer(new_words=msg, model_type="encoder")
        self.rnn_ae_model.expand_tokenizer(new_words=msg, model_type="decoder")
        reply_msg = self.rnn_ae_model.generate_sequence(input_question=msg, sequence_len=100, target_sequence_len=100)
        return reply_msg
    
    def _generate_reaction_img_(self, emotion_cll):
        
        generated_images_folder = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\bot\\generated_images_folder"
        self.gif_path = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\bot\\generated_images_folder\\emotion.gif"

        if not os.path.exists(generated_images_folder):
            os.mkdir(generated_images_folder)

        plt.style.use("dark_background")
        fig, axis = plt.subplots()
        
        random_idx = np.random.randint(0, self.em2img_model.hiden_dim[emotion_cll].shape[0])
        random_enc_sample = self.em2img_model.hiden_dim[emotion_cll][random_idx]
        walking_vector = random_enc_sample
        
        images = []
        images_number = 100
        for image_number in range(images_number):
            
            image_path = os.path.join(generated_images_folder, f"image_{image_number}.png")
            walking_vector += np.random.normal(0, 0.12, 2)
            expand_label = np.expand_dims(walking_vector, axis=0)

            gen_image = self.em2img_model.decoder.predict(expand_label)[0]
            gen_image += np.random.normal(0, 0.12, self.em2img_model.params_json["input_shape"])
            gen_image *= 255
            cv2.imwrite(image_path, gen_image)
        
        for image_path in os.listdir(generated_images_folder):
            
            image_path = os.path.join(generated_images_folder, image_path)
            image = Image.open(image_path)
            images.append(image)
        
        images[0].save(self.gif_path, save_all=True, append_images=images, optimize=False, duration=100, loop=0)
    
   

    
if __name__ == "__main__":
    bot = BotCore()
    
