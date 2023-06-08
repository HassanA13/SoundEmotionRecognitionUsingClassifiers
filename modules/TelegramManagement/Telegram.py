import telebot
from decouple import config
import os
import numpy as np
import time
import random
import sys
import helper as telegramHelper
sys.path.insert(0, os.getcwd())
import parameters as para 

class Telegram:
    def __init__(self) -> None:
        # Initialize the bot
        BOT_TOKEN = config('BOT_TOKEN')
        self.bot = telebot.TeleBot(BOT_TOKEN)
        # Load the model and print "ready" message
        # self.rec = get_classifier()
        # self.bot.send_message('hi')
        self.rec = para.classifiers[0].get_classifier_through_randomized_search_cv()


    # Function to process voice messages


    def process_voice_message(self,message, destination_file_path):
        self.bot.reply_to(message, f"Classifier Name: {para.classifiers[0].__name__}")

        telegramHelper.getWavDetail(destination_file_path)
        # emotion, score = predict(destination_file_path)
        emotion = para.classifiers[0].predict(destination_file_path)
        # score = self.rec.test_score()
        score = self.rec.best_score_
        # print(emotion)
        # return
        # score = 10 # , rec.test_score()
        # rec.predict_proba()
        # Rename the file based on the predicted emotion and the user's username (if available)

        # new_filename = f"{message.from_user.username}_{emotion}_{destination_file_path}" if message.from_user.username else f"{emotion}_{destination_file_path}"
        # os.rename(destination_file_path,os.path.join('temp', new_filename))

        # Reply to the user with the predicted emotion and score
        self.bot.reply_to(message, f"The score is: {score}")
        self.bot.reply_to(message, f"The emotion is: {emotion}")

        # Respond with an emoji based on the predicted emotion
        emoji_dict = {
            'neutral': '🙂',
            'happy': '😄',
            'sad': '🥺',
            'angry': '😡',
            'ps': '😲',
            'calm': '😌',
            'fear': '😱',
            'disgust': '🤢',
            'boredom': '🥱'
        }
        # default to neutral face if emotion is unknown
        emoji = emoji_dict.get(emotion, '😐')
        self.bot.reply_to(message, emoji)

    def run(self):
        print('Model is ready')
        # Handle voice and audio messages
        @self.bot.message_handler(func=lambda message: message.voice.mime_type == 'audio/ogg', content_types=['voice'])
        def handle_audio_message(message):
            # bot.reply_to(message,message)
            file_info = self.bot.get_file(message.voice.file_id)

            # Download the voice file
            self.bot.reply_to(message, 'The voice is being processed. Please wait.')
            downloaded_file = self.bot.download_file(file_info.file_path)
            # bot.reply_to(message, '1')

            # Save the file to disk
            received_file_path = os.path.join(os.path.dirname(
                __file__), 'temp', f"{message.voice.file_id}.ogg")
            destination_file_path = os.path.join(os.path.dirname(
                __file__), 'temp', f"{int(time.time())}_{int(random.random() * 9999999999)}.wav")
            with open(received_file_path, 'wb') as new_file:
                new_file.write(downloaded_file)
            destination_after_reduce_noise_file_path = os.path.join(os.path.dirname(
                __file__), 'temp', f"{int(time.time())}_after_reduce_noise_{int(random.random() * 9999999999)}.wav")
            with open(received_file_path, 'wb') as new_file:
                new_file.write(downloaded_file)

            telegramHelper.ogg2wav(received_file_path,destination_file_path)
            telegramHelper.reduce_noise(destination_file_path,destination_after_reduce_noise_file_path)
            telegramHelper.scale_amplitude(destination_file_path,destination_file_path,0.3)
            telegramHelper.scale_amplitude(destination_after_reduce_noise_file_path,destination_after_reduce_noise_file_path,0.3)


            self.process_voice_message(message, destination_file_path)
            self.process_voice_message(message, destination_after_reduce_noise_file_path)


        @self.bot.message_handler(func=lambda message: message.document.mime_type == 'audio/x-wav', content_types=['document'])
        def handle_wav_file(message):
            # Process the WAV file here
            # bot.reply_to(message,message)
            file_info = self.bot.get_file(message.document.file_id)

            # Download the voice file
            self.bot.reply_to(message, 'The voice is being processed. Please wait.')
            downloaded_file = self.bot.download_file(file_info.file_path)

            received_file_path = os.path.join(os.path.dirname(__file__), 'temp', f"{message.document.file_name}")
            # destination_file_path   = os.path.join(os.path.dirname(__file__), 'temp', f"{int(time.time())}_{int(random.random() * 9999999999)}.wav")
            destination_after_reduce_noise_file_path = os.path.join(os.path.dirname(__file__), 'temp', f"{int(time.time())}_after_reduce_noise_{int(random.random() * 9999999999)}.wav")
            with open(received_file_path, 'wb') as new_file:
                new_file.write(downloaded_file)
            with open(received_file_path, 'wb') as new_file:
                new_file.write(downloaded_file)

            telegramHelper.reduce_noise(received_file_path,destination_after_reduce_noise_file_path)
            telegramHelper.scale_amplitude(received_file_path,received_file_path,0.3)
            telegramHelper.scale_amplitude(destination_after_reduce_noise_file_path,destination_after_reduce_noise_file_path,0.3)

            self.process_voice_message(message, received_file_path)
            self.process_voice_message(message, destination_after_reduce_noise_file_path)


        # Handle document messages
        @self.bot.message_handler(content_types=['document'])
        def handle_document_message(message):
            # bot.reply_to(message,message)
            self.bot.reply_to(message, 'Only audio files are supported at this time.')

        # Start polling for messages
        self.bot.polling()
