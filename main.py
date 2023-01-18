from aiogram.utils import executor
from create_bot import dp
from handlers.users_part import register_handler_users


async def on_startup(_):
    print("Bot online")
    #sqlite_db.sql_start()

register_handler_users(dp)

executor.start_polling(dp, skip_updates=True, on_startup=on_startup)
