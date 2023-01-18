from aiogram.types.reply_keyboard import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove


but_start_working = KeyboardButton("/start_working")
but_watch_examples = KeyboardButton("/watch_examples")
greeting_keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but_start_working, but_watch_examples]])

but_watch_more_examples = KeyboardButton("/watch_more_examples")
watching_examples_keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but_start_working, but_watch_more_examples]])