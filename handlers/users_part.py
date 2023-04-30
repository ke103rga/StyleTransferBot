import aiofiles
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram import types
from aiogram.dispatcher import Dispatcher
from create_bot import bot
from keyboards import client_kb
from database import database
from style_transfer.style_transfer import syntez_image
from create_work_directory import images_dir
import stat
import os
import aiohttp


class FSM(StatesGroup):
    greeting = State()
    examples_showing = State()
    loading_content_image = State()
    choosing_sourse_of_style_image = State()
    loading_castom_style_image = State()
    viewing_gallery = State()
    loading_gallery_style_image = State()
    setting_con_st_ratio = State()
    image_synthesing = State()
    loading_result_to_gallery = State()


users_data = {}


async def command_start(message: types.Message):
    await FSM.greeting.set()
    text = "Hello, You've just visited a bot that can do unbelievable things with different photos and images.\n" \
           "Are you already familiar with them or prefer to watch some examples firstly?"
    await bot.send_message(chat_id=message.from_user.id, text=text, reply_markup=client_kb.greeting_keyboard)


async def command_examples_showing(message: types.Message):
    await FSM.examples_showing.set()
    await database.show_gallery(bot, message, 1)
    await bot.send_message(message.from_user.id,
                           text="Ok, now let's create something as exciting \n"
                                "as image you've just seen.",
                           reply_markup=client_kb.start_working_keyboard)


async def command_start_working(message: types.Message):
    explain_text = "Ok. Let me explain some details.\n" \
                   "It's up to you to load photo whose content you wish to display\n" \
                   "and photo whose style you want to use.\n" \
                   "I'll show you an example."
    ask_text = "First of all load content image you can send a photo or a link of it."
    user_id = message.from_user.id
    # Creating temporary storage of user's information for working with style transfer
    users_data[user_id] = {"content_image": None, "style_image": None, "ratio": 1, "style_id": None}
    await bot.send_message(chat_id=message.from_user.id, text=explain_text)
    await database.show_gallery(bot, message, 1)
    await FSM.loading_content_image.set()
    await bot.send_message(chat_id=message.from_user.id, text=ask_text)


async def load_content_image(message: types.Message):
    # Trying to load the content image that user's just send
    # And insert information about location of particular user's data
    load_status = await load_image(message, images_dir, "content")
    # Asking about style image
    if load_status == "OK":
        await bot.send_message(message.from_user.id,
                               text="Ok. Now let's load a style image.\n"
                                    "If you have your own idea about which style image do you want to use, "
                                    "please tap respective key."
                                     "Or i can show you our gallery and you can choose a style image from there.",
                               reply_markup=client_kb.style_download_way_keyboard)
        await FSM.choosing_sourse_of_style_image.set()
    # Asking about more attempts of sending content image
    else:
        await bot.send_message(message.from_user.id,
                               text="There are some issues with that picture\n"
                                    "please choose another one and send it.")


async def command_use_my_own_image(message: types.Message):
    await bot.send_message(message.from_user.id,
                           text="Ok, so load your image the same way as content image.\n"
                                "Send me a photo or a link of it.")
    # Inserting the information that user preferred to use his own photo
    # and there isn't it in database.
    users_data[message.from_user.id]["style_source"] = -1
    await FSM.loading_castom_style_image.set()


async def command_show_gallery(message: types.Message):
    # Sending explanation about system of choosing style image from gallery
    await bot.send_message(message.from_user.id,
                           text="Now just tap the key under the image that you like the most.")
    await database.show_gallery(bot, message, 5)
    # Sending explanation for case if user hadn't chosen anything
    await bot.send_message(message.from_user.id,
                           text="If you didn't like anything of our examples"
                                "you still can load your own image.",
                           reply_markup=client_kb.use_own_image_keyboard)
    await FSM.loading_gallery_style_image.set()


async def load_castom_style_image(message: types.Message):
    # Trying to load the content image that user's just send
    # And insert information about location of particular user's data
    loading_status = await load_image(message, images_dir, "style")
    # Asking for final style transfer's params tuning
    if loading_status == "OK":
        await asking_for_ratio(message.from_user.id)
        await FSM.setting_con_st_ratio.set()
    # Asking about more attempts of sending style image
    else:
        await bot.send_message(message.from_user.id,
                               text="There are some issues with that picture\n"
                                    "please choose another one and send it.")


async def load_gallery_style_image(callback_query: types.CallbackQuery):
    """
    The function that hand callback queries from users who preferred to use gallery style images
     it gets id of style image from database logs it into users data and loads the image from database
     to user's temporary folder
    :param callback_query: types.CallbackQuery, from choosing the style image from gallery
    :return: None
    """
    user_id = callback_query.from_user.id
    # getting the id of style image from database
    style_id = int(callback_query.data.replace("use ", ""))
    # Logging information
    users_data[user_id]["style_id"] = style_id
    users_data[user_id]["style_image"] = f"{images_dir}\\user_{user_id}\\style_image.jpg"
    # Loading the image from database to user's temporary folder
    await database.load_gallery_style_image(style_id, images_dir, user_id)
    await asking_for_ratio(user_id)
    await FSM.setting_con_st_ratio.set()


async def setting_con_st_ratio(message: types.Message):
    # Checking if ratio is valid
    if message.text.isdigit() and 1 <= int(message.text) <= 5:
        try:
            # Logging information about rayio
            style_weight = int(message.text)
            users_data[message.from_user.id]["style_weight"] = style_weight
            await FSM.image_synthesing.set()
            # synthesizing new image
            user_id = message.from_user.id
            await syntez_image(user_id=user_id, user_data=users_data[user_id])
            # sanding the result
            user_dir = f"{images_dir}\\user_{user_id}"
            photo = types.input_file.InputFile(f"{user_dir}\\result_image.jpg")
            await bot.send_photo(user_id,
                                 photo=photo,
                                 caption="The result of style transfering.")
            # Asking about desire to save result in gallery
            await bot.send_message(user_id,
                                   "Do you want to put the result into our gallery?\n"
                                   "Please choose respective variant.",
                                   reply_markup=client_kb.agreement_keyboard)
            await FSM.loading_result_to_gallery.set()
        except Exception:
            await bot.send_message(message.from_user.id,
                                   text="Some problems have arisen wit the dowloading your images.\n"
                                        "Please make sure you have done everything according to the instructions\n"
                                        "or just choose other photos.\n"
                                        "Send me a content photo or a link of it again.")
            await FSM.loading_content_image.set()
    # Asking for more attempts
    else:
        await bot.send_message(message.from_user.id,
                               text="It's necessary to enter the integer number"
                               "between 1 and 5 including.")


async def handler_for_synthesing_period(message: types.Message):
    # Handler for situations when user tries to continue conversation while result still in syntheses process
    await bot.send_message(message.from_user.id,
                           text="Please wait for some minutes."
                                "It's a long process to redraw image in different style.")


def clean_thrash(path):
    dirlist=get_dirlist(path)
    for f in dirlist:
        fullname=os.path.join(path,f)
        if fullname == os.path.join(path,"thrash.txt"):
            os.chmod(fullname , stat.S_IWRITE)
            os.remove(fullname)
        if os.path.isdir(fullname):
            clean_thrash(fullname)


async def loading_result_to_gallery(message: types.Message):
    answer = message.text
    user_id = message.from_user.id
    if answer == "Yes":
        await database.load_to_gallery(bot, images_dir, user_id, users_data[user_id].get("style_id", -1))
        await bot.send_message(user_id,
                               "Thank you\n That's all\n"
                               "If you want to try some other images,"
                               "please tap the 'start' key",
                               reply_markup=client_kb.restart_keyboard)
    elif answer == "No":
        await bot.send_message(user_id,
                               "Ok, it's your choice.\n"
                               "If you want to try some other images,"
                               "please tap the 'start' key",
                               reply_markup=client_kb.restart_keyboard)
    else:
        await bot.send_message(user_id,
                               "I haven't understood,\n "
                               "so I won't load your result to gallery,\n"
                               "but if you want to try some other images,"
                               "please tap the 'start' key",
                               reply_markup=client_kb.restart_keyboard)
    user_dir = f"{images_dir}\\user_{user_id}"
    # Deleting the temporary folder with user's images
    # if user agreed they were already loaded to database
    if os.path.isdir(user_dir):
        os.remove(user_dir)
    # Deleting temporary storage of user's information for working with style transfer
    del users_data[user_id]


async def asking_for_ratio(user_id):
    explain_text = "Excellent!\nThe only question is left to solve is:\n" \
                   "how strongly convey style.\n"
    await bot.send_message(user_id,
                           text=explain_text)
    ask_text= "Ok, now enter the number between 1 and 5 including."
    await bot.send_message(user_id,
                           text=ask_text)


async def load_image(message: types.Message, images_dir, image_type):
    user_id = message.from_user.id
    user_dir = f"{images_dir}\\user_{user_id}"
    # Checking if folder  with user's images exists
    if not os.path.isdir(user_dir):
        os.mkdir(user_dir)
    destination_file = f"{user_dir}\\{image_type}_image.jpg"
    if message.content_type == "photo":
        # Downloading the photo
        await message.photo[-1].download(destination_file=destination_file)
    elif message.content_type == "text":
        # Checking if message text is link
        if not "http" in message.text:
            await bot.send_message(user_id, text="Please, send a link to a photo or just a photo")
            return "Failed loading"
        # Downloading image from link and saving it in user's folder
        async with aiohttp.ClientSession() as session:
            async with session.get(message.text) as resp:
                if resp.status == 200:
                    f = await aiofiles.open(destination_file, mode='wb')
                    await f.write(await resp.read())
                    await f.close()
                else:
                    return "Failed loading"
    # Inserting information about file location
    users_data[user_id][f"{image_type}_image"] = destination_file
    return "OK"


async def restart(message: types.Message):
    await bot.send_message(message.from_user.id,
                           "It seems you haven't finished our last conversation\n"
                           "If you want to restart plese enter the '/start' command",
                           reply_markup=client_kb.restart_keyboard)


def register_handler_users(dp: Dispatcher):
    dp.register_message_handler(command_start, commands=["start"], state="*")
    dp.register_message_handler(command_examples_showing, commands=["watch_examples"], state=FSM.greeting)
    dp.register_message_handler(command_start_working, commands=["start_working"],
                                state=[FSM.greeting, FSM.examples_showing])
    dp.register_message_handler(command_use_my_own_image, commands=["use_my_own_image"],
                                state=[FSM.choosing_sourse_of_style_image, FSM.loading_gallery_style_image])
    dp.register_message_handler(command_show_gallery, commands=["show_gallery"], state=FSM.choosing_sourse_of_style_image)
    dp.register_message_handler(load_content_image, state=FSM.loading_content_image, content_types=["photo", "text"])
    dp.register_message_handler(load_castom_style_image, state=FSM.loading_castom_style_image, content_types=["photo", "text"])
    dp.register_message_handler(setting_con_st_ratio, state=FSM.setting_con_st_ratio)
    dp.register_message_handler(handler_for_synthesing_period, state=FSM.image_synthesing)
    dp.register_callback_query_handler(load_gallery_style_image,
                                       lambda callback_query: callback_query.data.startswith('use'),
                                       state=FSM.loading_gallery_style_image)
    dp.register_message_handler(loading_result_to_gallery, state=FSM.loading_result_to_gallery)
    dp.register_message_handler(restart, state="*")
