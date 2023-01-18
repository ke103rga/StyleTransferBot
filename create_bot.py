from aiogram import Bot
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage


storage = MemoryStorage()

token = "5874033941:AAHwijw6f1pdnw-ROa2iczP_IQXLKZbPOZA"

bot = Bot(token=token)
dp = Dispatcher(bot, storage=storage)