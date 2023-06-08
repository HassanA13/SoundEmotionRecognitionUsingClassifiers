import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd()+"\modules\TelegramManagement")
from modules.TelegramManagement.Telegram import Telegram


Telegram().run()