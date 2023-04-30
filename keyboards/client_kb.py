from aiogram.types.reply_keyboard import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


but_start_working = KeyboardButton("/start_working")
but_watch_examples = KeyboardButton("/watch_examples")
greeting_keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but_start_working, but_watch_examples]], one_time_keyboard=True)

start_working_keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but_start_working]], one_time_keyboard=True)

but_own_style_image = KeyboardButton("/use_my_own_image")
use_own_image_keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but_own_style_image]], one_time_keyboard=True)

but_show_gallery = KeyboardButton("/show_gallery")
style_download_way_keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but_own_style_image, but_show_gallery]])

but_start = KeyboardButton("/start")
restart_keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but_start]])

but_agreement = KeyboardButton("Yes")
but_disagreement = KeyboardButton("No")
agreement_keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but_agreement, but_disagreement]], one_time_keyboard=True)

# but_use_style_image = InlineKeyboardMarkup().add(InlineKeyboardButton("Use as a style image", callback_data=f"use {elem[1]}"))