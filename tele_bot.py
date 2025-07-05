import asyncio
import json
import requests
from telethon import TelegramClient, events

from asyncio import Queue

from twitter_cli import check_response
from llm import generate_response, llm_with_tools
from tools import flight_search, handle_feedback, hotel_search, fetch_user_flight_information, lookup_policy, cancel_ticket
import os
from dotenv import load_dotenv
load_dotenv()

# Your Telegram API credentials
api_id = os.environ.get('API_ID')
api_hash = os.environ.get('API_HASH')
phone_number = os.environ.get('PHONE_NUMBER') 
group_chat_id = int(os.environ.get('GROUP_CHAT_ID'))
bot_token = os.environ.get('BOT_TOKEN')


# Initialize the client
client = TelegramClient('newbot', api_id, api_hash).start(phone=phone_number)
# client = TelegramClient('new_bot', api_id, api_hash).start(bot_token=bot_token)

# Function to handle new messages
@client.on(events.NewMessage(chats=[group_chat_id]))
async def new_message_handler(event):
    sender = await event.get_sender()
    message_text = event.message.message
    chat_id = event.chat_id
    message_id = event.message.id
    print("GET MESSAGE: ", event.raw_text.lower())
    
    # Check if the message contains "flight"
    # if "flight" in event.raw_text.lower():
    #     sender = await event.get_sender()  # Get message sender
        # response = check_response(message_text)
    # response = generate_response(message_text)
    tools = [flight_search, hotel_search, cancel_ticket, lookup_policy, fetch_user_flight_information, handle_feedback]
    try:
        response = llm_with_tools(message_text, tools)
    except Exception as e:
        print("Error: ", e)
        response = "The system is under maintenance, please try again later."

    print("Response: ", response)
    # Respond to the sender in the same chat
    # await event.reply(response)
    if not "UNRELATED" in response:
        await event.reply(response)
    else:
        print("Unrelated messages!")


# Main function
async def main():
    # Directly await the send_telegram_message coroutine
    await send_telegram_message(client, -4710560773, "Test message", 15959)


async def send_telegram_message(client, chat_id, text, reply_to):
    print("CHECKPONT: ", chat_id, text, reply_to )
    
    await client.send_message(
        chat_id,
        text,
        reply_to=reply_to
    )
    print(f"Replied to message ID {reply_to} in chat {chat_id}")
    await asyncio.sleep(2)
def get_history():
    data = requests.get(f"https://api.telegram.org/bot{bot_token}/getUpdates")
    return data
# Start the client
if __name__ == "__main__":
    with client:
        # client.loop.run_until_complete(client.run_until_disconnected())
        client.run_until_disconnected()
    # noti_tele("Hello Andy", 15799)
    # start()
    # bot = telegram.Bot(token=bot_token)
    # bot.send_message(chat_id=group_chat_id, text='USP-Python has started up!')
    # with client:
    #         client.loop.run_until_complete(
    #             send_telegram_message(client, 6116391670, "Test message", 209)
    #         )
    # data = get_history()
    # print(data.content)