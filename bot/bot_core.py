import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2


from PIL import Image
from matplotlib.animation import FuncAnimation
from models import VarEncoder
from models import RNN
from matplotlib.animation import FuncAnimation
from telegram.ext import Application, filters, ContextTypes
from telegram.ext import MessageHandler, CommandHandler
from telegram import Update

class BotCore:

    def __init__(self) -> None:
        

        
        self.seq2em_model = RNN(filepath="C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\RNN.json")
        self.em2img_model = VarEncoder(filepath="C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\VarAutoEncoder.json")

        self.seq2em_model.load_weights()
        self.em2img_model.load_weights()


        self.saved_images_folder = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\save_images_folder"
        self.saved_voices_folder = "C:\\Users\\1\\Desktop\\TelegramAIBotProject\\bot\\save_images_folder"
        self.app_builder = Application.builder()
        self.app_builder.token("7405563731:AAHhDPtAHWXzi-ZCRh-F-uWMBtBZSyrv3gg")
        self.application = self.app_builder.build()

        self.start_handler = CommandHandler(command="start", callback=self.show_functional)
        self.images_handler = MessageHandler(filters=filters.PHOTO, callback=self.save_image)
        self.voice_handler = MessageHandler(filters=filters.VOICE, callback=self.voice_image)
        
        
        
        all_handlers = [self.start_handler, self.msg_handler]
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
        
            
            
    def _extract_emotion_label_(self, msg):

        
        self.seq2em_model.expand_tokenizer(new_words=msg)

        token_list = self.seq2em_model.tokenizer.texts_to_sequences([msg])[0]
        token_list = np.expand_dims(np.asarray(token_list, dtype="int64"), axis=0)

        emotion_label = np.argmax(self.seq2em_model.model.predict(token_list)[0])
        emotion_cll = self.seq2em_model.rnn_params["classes_discription"][str(emotion_label)]
        return (emotion_label, emotion_cll)
    

    async def emotion_classification (self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        
        message_text = update.message.text.split()
        _, emotion_cll = self._extract_emotion_label_(msg=message_text)
        self._generate_reaction_img_(emotion_cll=emotion_cll)

        await update.message.reply_text(f"Its look like you feel {emotion_cll} right know!!!")
        await update.message.reply_animation(animation=open(self.gif_path, "rb"))


    async def show_functional(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Hello world")
    
    async def save_msg(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        
        image_number = len(os.listdir(self.images_handler)) + 1
        image_path = os.path.join(self.saved_images_folder, f"image{image_number}.png")
        
        image = await update.message.effective_attachment[-1].get_file()
        await update.message.d
        
    
    
if __name__ == "__main__":
    bot = BotCore()
    
