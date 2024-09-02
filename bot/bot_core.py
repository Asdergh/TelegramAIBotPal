import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import json as js
import tensorflow.keras.backend as K

from PIL import Image
from matplotlib.animation import FuncAnimation
from models.conv_var_ae import VarEncoder
from models.rnn import RNN
from models.rnn_ae import RNN_AE
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

        self.start_handler = CommandHandler(command="start", callback=self.show_functional)
        self.images_handler = MessageHandler(filters=filters.PHOTO, callback=self.save_image)
        self.voice_handler = MessageHandler(filters=filters.VOICE, callback=self.save_image)
        self.text_handler = MessageHandler(filters=filters.TEXT, callback=self.emotion_classification)
        
        
        all_handlers = [self.start_handler, self.images_handler, self.voice_handler, self.text_handler]
        for handler in all_handlers:
            self.application.add_handler(handler)
        self.application.run_polling(poll_interval=3)
        
        
    
    
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
        
            
            
    def _generate_answer(self, msg):

        
        self.seq2em_model.expand_tokenizer(new_words=msg.split())
        self.rnn_ae_model.expand_tokenizer(new_words=msg.split())

        classification_tokens = self.seq2em_model.tokenizer.texts_to_sequences([msg.split()])[0]
        classification_tokens = np.expand_dims(np.asarray(classification_tokens, dtype="int64"), axis=0)
        
        emotion_label = np.argmax(self.seq2em_model.model.predict(classification_tokens)[0])
        emotion_cll = self.seq2em_model.rnn_params["classes_discription"][str(emotion_label)]
        
        encoder_input = self.rnn_ae_model.encoder_tokenizer.texts_to_sequences([msg])[0]
        encoder_input = np.asarray(encoder_input)
        print(encoder_input.shape)
        #answer_msg = self.rnn_ae_model.generate_sequence(input_question=msg, sequence_len=100, target_sequence_len=40)
        
        return (emotion_cll, answer_msg)
    

    async def emotion_classification (self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        
        message_text = update.message.text
        emotion_cll, generated_answer = self._generate_answer(msg=message_text)
        self._generate_reaction_img_(emotion_cll=emotion_cll)

        await update.message.reply_text(f"Well as i can see you are in {emotion_cll} mood." + generated_answer)
        await update.message.reply_animation(animation=open(self.gif_path, "rb"))


    async def show_functional(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Hello world")
    
    async def save_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        
        image_number = len(os.listdir(self.images_handler)) + 1
        image_path = os.path.join(self.saved_images_folder, f"image{image_number}.png")
        
        image = await update.message.effective_attachment[-1].get_file()
        await update.message.d
    
    async def save_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pass
        
    
    
if __name__ == "__main__":
    bot = BotCore()
    
