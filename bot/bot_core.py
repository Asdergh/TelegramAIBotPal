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

        self.images_folder = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\bot\\images_folder"
        self.app_builder = Application.builder()
        self.app_builder.token("7405563731:AAHhDPtAHWXzi-ZCRh-F-uWMBtBZSyrv3gg")
        self.application = self.app_builder.build()

        self.start_handler = CommandHandler(command="start", callback=self.show_functional)
        self.images_handler = MessageHandler(filters=filters.PHOTO, callback=self.save_image)
        self.text_handler = MessageHandler(filters=filters.TEXT, callback=self.emotion_classification)
        
        
        
        all_handlers = [self.start_handler, self.images_handler, self.text_handler]
        for handler in all_handlers:
            self.application.add_handler(handler)
        self.application.run_polling(poll_interval=3)
        
        
    
    
    def _generate_reaction_img_(self, emotion_cll):
        
        images_folder = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\bot\\images_folder"
        self.gif_path = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\bot\\images_folder\\emotion.gif"

        if not os.path.exists(images_folder):
            os.mkdir(images_folder)

        plt.style.use("dark_background")
        fig, axis = plt.subplots()
        
        random_idx = np.random.randint(0, self.em2img_model.hiden_dim[emotion_cll].shape[0])
        random_enc_sample = self.em2img_model.hiden_dim[emotion_cll][random_idx]
        walking_vector = random_enc_sample
        
        images = []
        images_number = 100
        for image_number in range(images_number):
            
            image_path = os.path.join(images_folder, f"image_{image_number}.png")
            walking_vector += np.random.normal(0, 0.12, 2)
            expand_label = np.expand_dims(walking_vector, axis=0)

            gen_image = self.em2img_model.decoder.predict(expand_label)[0]
            gen_image += np.random.normal(0, 0.12, self.em2img_model.params_json["input_shape"])
            gen_image *= 255
            cv2.imwrite(image_path, gen_image)
        
        for image_path in os.listdir(images_folder):
            
            image_path = os.path.join(images_folder, image_path)
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
    
    async def save_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        
        image_file = await update.message.effective_attachment[-1].get_file()
        image_file = await image_file.download_to_drive(custom_path=f"C:\\Users\\1\\Desktop\\PersonalFriendProject\\new_photo\\image_{self.image_number}.png")
        image_file = str(image_file)
    
    
if __name__ == "__main__":
    bot = BotCore()
    
