from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram import types
from aiogram.dispatcher import Dispatcher
from create_bot import bot, dp
from keyboards import greeting_keyboard, watching_examples_keyboard


class FSM(StatesGroup):
    greeting = State()
    start_working = State()
    examples_showing = State()
    loading_content_image = State()
    loading_castom_style_image = State()
    viewing_gallerie = State()
    loading_gallerie_style_image = State()
    setting_con_st_ratio = State()
    image_synthesing = State()
    loading_result_to_gallery = State()


async def greeting(message: types.Message):
    await FSM.greeting.set()
    text = "Hello, You've just visited a bot that can do unbelievable things with different photos and images.\n" \
           "Are you already familiar with them or prefer to watch some examples firstly?"
    await bot.send_message(chat_id=message.from_user.id, text=text, reply_markup=greeting_keyboard)


async def command_examples_showing(message: types.Message):
    await FSM.examples_showing.set()
    await bot.send_message(chat_id=message.from_user.id,
                           text="I will send you some examples in the future",
                           reply_markup=watching_examples_keyboard)


async def command_start_working(message: types.Message):
    pass


def register_handler_users(dp: Dispatcher):
    dp.register_message_handler(greeting, commands=["start"], state=None)
    dp.register_message_handler(command_examples_showing, commands=["watch_examples"], state=FSM.greeting)