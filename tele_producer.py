
from telethon import TelegramClient, events
from kafka import KafkaProducer
import json
import os
from dotenv import load_dotenv
load_dotenv()

# Telegram credentials
api_id = os.environ.get('API_ID')
api_hash = os.environ.get('API_HASH')
phone_number = os.environ.get('PHONE_NUMBER')
group_chat_id = int(os.environ.get('GROUP_CHAT_ID'))
print(group_chat_id)

# Kafka setup
KAFKA_BROKER = 'localhost:9092'
TOPIC = 'telegram_messages'

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # Serialize messages as JSON
)

# Initialize the Telegram client
client = TelegramClient('producer_session', api_id, api_hash).start(phone=phone_number)
@client.on(events.NewMessage(chats=[group_chat_id]))
async def handle_new_message(event):
    message_data = {
        'chat_id': event.chat_id,
        'sender_id': event.sender_id,
        'id': event.message.id,  # Include the message ID
        'text': event.raw_text,
        'timestamp': event.date.isoformat()
    }
    producer.send(TOPIC, message_data)
    print(f"Sent message to Kafka: {message_data}")

# Start the producer client
async def main():
    print("Producer is running...")
    await client.run_until_disconnected()

if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(main())
