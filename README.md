This will you personal telegram bot powerd by AI. It will handle text, iamge and even audio generation to entertain you and make you more happier.
Also in next versions it will be able to make image transfer, and image generation from your text reguests.

![brain_image](https://github.com/user-attachments/assets/214d99b9-8f9d-4c04-af72-bd1a536e2c06)

Curent possibilities: 
  1. Emotino classification from text request
  2. Generate face image as a call for your text request considering emotion in your message

Possibilities add shudule:
  1. Full text generation to talk with users
  2. Full image handling to make image generation more interesting
  3. Audio processing from text to generation audio sequences as a call for your request(to talk with you)
  4. Text Extraction from images, so you would be able to talk with bot thought hand writen text

Next probable possibilities (depends on work speed):
  1. Automatize your work env with my bot
  2. Automatize your chats workspace with my bot
  3. Possiblly i will add web version

Bot application scheme (only first test look):

![Диаграмма без названия drawio (2)](https://github.com/user-attachments/assets/6c987593-6648-4e0d-a7c6-647d7f22f973)

. Ai api:

  . Models module contain all models that take place in analization processes of the bot
  . Custom Callbacks is a module with custom callbakcs for models trainig processes analization 
  . Layer is a module with custom layers for models in Models module 

. Bot Api:
  . For know i have only basic telegram bot api realizatio to send messages, analize them. Also i have write some function to handle request realeted with AI tasks
    such as images NST task, image generation task. I will add some features and little functionality to manipulate with my bot and make own applications based on 
    requests to my bot :)


