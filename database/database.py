from aiogram import types
from aiogram import Bot
import asyncio
from getpass import getpass
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import mysql.connector
from mysql.connector import connect, Error
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import base64


def create_database(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as e:
        print(f"The error '{e}' occurred")


def create_tables(connection):
    images_tables = ["content_images", "style_images", "result_images"]
    create_images_table_query = "CREATE TABLE table_name(" \
                                "id INT PRIMARY KEY AUTO_INCREMENT," \
                                "caption VARCHAR(30)," \
                                "author VARCHAR(30)," \
                                "image MEDIUMBLOB NOT NULL)"
    create_users_table_query = "CREATE TABLE users(" \
                               "id INT PRIMARY KEY AUTO_INCREMENT," \
                               "user_name VARCHAR(30))"
    create_gallery_table_query = "CREATE TABLE gallery(" \
                                 "id INT PRIMARY KEY AUTO_INCREMENT," \
                                 "content_id INT," \
                                 "style_id INT," \
                                 "result_id INT," \
                                 "author_id INT," \
                                 "FOREIGN KEY (content_id) REFERENCES content_images (id)," \
                                 "FOREIGN KEY (style_id) REFERENCES style_images (id)," \
                                 "FOREIGN KEY (result_id) REFERENCES result_images (id)," \
                                 "FOREIGN KEY (author_id) REFERENCES users (id))"
    cursor = connection.cursor()
    try:
        for image_table in images_tables:
            query = create_images_table_query.replace("table_name", image_table)
            cursor.execute(query)
        cursor.execute(create_users_table_query)
        cursor.execute(create_gallery_table_query)
        print("All tables were created successfully")
    except Error as e:
        print(f"The error '{e}' occurred")


def create_connection_to_server(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


def create_connection_to_database(host_name, user_name, user_password, database="style_transfer_gallery"):
    connection = None
    try:
        connection = connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=database
        )
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


async def show_gallery(bot: Bot, message: types.Message, limit=1):
    with create_connection_to_database("localhost", "root", "Ege948489_?") as con:
        cursor = con.cursor()
        # Creating a related table and load images of all types
        query = "select content_images.image, style_images.image, style_images.id,  result_images.image " \
                "from gallery " \
                "inner join content_images on content_id = content_images.id " \
                "inner join style_images on style_id = style_images.id " \
                "inner join result_images on result_id = result_images.id " \
                "order by rand() " \
                f"limit {limit}"
        cursor.execute(query)
        gallery_images = cursor.fetchall()
        for content, style, style_id, result in gallery_images:
            # Creating a message that contains 3 images
            media = types.MediaGroup()
            media.attach_photo(types.InputMediaPhoto(types.InputFile(BytesIO(content))), "Content image")
            media.attach_photo(types.InputMediaPhoto(types.InputFile(BytesIO(style))), "Style image")
            media.attach_photo(types.InputMediaPhoto(types.InputFile(BytesIO(result))), "Result image")
            # It's just a single example and contains explanations
            if limit == 1:
                await bot.send_media_group(chat_id=message.from_user.id,
                                           media=media)
                await bot.send_message(chat_id=message.from_user.id,
                                       text="The first two images are content and style images as you can see.\n"
                                            "You need to load both of them and a few minutes later after it\n"
                                            "I will send you a wondering result like a third image.")
            # It's a case of showing of part of gallery with suggestions to use style images for
            # user's creative
            else:
                await bot.send_media_group(chat_id=message.from_user.id,
                                           media=media)
                keyboard = InlineKeyboardMarkup().add(InlineKeyboardButton("Use as a style image",
                                                                           callback_data=f"use {style_id}"))
                await bot.send_message(message.from_user.id,
                                       text="Do you want to use that style image for your creation?",
                                       reply_markup=keyboard)


async def load_gallery_style_image(style_id, images_dir, user_id):
    user_dir = f"{images_dir}\\user_{user_id}"
    with create_connection_to_database("localhost", "root", "Ege948489_?") as con:
        cursor = con.cursor()
        query = f"select image from style_images where id = {style_id}"
        cursor.execute(query)
        image = cursor.fetchone()[0]
        # Loading a style image from database and saving it to work directory
        with open(f"{user_dir}\\style_image.jpg", 'wb') as file:
            file.write(image)


async def load_to_gallery(bot, images_dir, user_id, style_id):
    with create_connection_to_database("localhost", "root", "Ege948489_?") as con:
        cursor = con.cursor()
        user_dir = f"{images_dir}\\user_{user_id}"
        if style_id > 0:
            # In that case style image was loaded from database and
            # there isn't necessity to load it back
            insert_ids = {"content": 0, "result": 0}
        else:
            insert_ids = {"content": 0, "style": 0, "result": 0}
        try:
            # Inserting all types of images in respective tables
            for image_type in insert_ids.keys():
                table_name = f"{image_type}_images"
                filepath = f"{user_dir}\\{image_type}_image.jpg"
                with open(filepath, "rb") as photo:
                    blob_photo = photo.read()
                query = f"insert into {table_name}(image) values(%s)"
                args = (blob_photo, )
                cursor.execute(query, args)
                insert_ids[image_type] = cursor.lastrowid
            # Inserting user as an author of result
            cursor.execute(f"insert into users(user_name) values({user_id})")
            author_id = cursor.lastrowid
            # Inserting all the information about particular work in gallery
            gallery_query = ("insert into gallery(content_id, style_id, result_id, author_id)"
                              "values(%s, %s, %s, %s)")
            if style_id > 0:
                gallery_args = (insert_ids["content"], style_id, insert_ids["result"], author_id)
            else:
                gallery_args = (insert_ids["content"], insert_ids["style"], insert_ids["result"], author_id)
            cursor.execute(gallery_query, gallery_args)
            # Saving the changes
            con.commit()
        except Error as e:
            print(f"The error '{e}' occurred")


def insert_image(table_name, filepath, caption=None, author=None):
    with create_connection_to_database("localhost", "root", "Ege948489_?") as con:
        cursor = con.cursor()
        with open(filepath, "rb") as photo:
            blob_photo = photo.read()
        query = f"insert into {table_name}(caption, image, author) values(%s, %s, %s)"
        args = (caption, blob_photo, author)
        cursor.execute(query, args)
        con.commit()


# with create_connection_to_database("localhost", "root", "Ege948489_?") as con:
#     cursor = con.cursor()
#     cursor.execute("insert into gallery(content_id, style_id, result_id)"
#                    "values(2, 3, 1)")
#     print()

# event_loop = asyncio.get_event_loop()
# tasks = [event_loop.create_task(load_gallery_style_image(3,
#                                                 "C:\\Users\\User1\\PythonProg\\PycharmProjects\\StyleTransferBot\\users_images",
#                                                 5196693208))]
# event_loop.run_until_complete(asyncio.wait(tasks))
# event_loop.close()

    # image = cursor.fetchall()[3][3]
    # image = Image.open(BytesIO(image)).convert('RGB')
    # plt.imshow(image)
    # plt.show()


