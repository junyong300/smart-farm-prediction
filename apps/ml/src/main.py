import asyncio
from asyncio.exceptions import CancelledError
import async_timeout
import aioredis
import json

STOPWORD = "STOP"

async def reader(channel: aioredis.client.PubSub, redis: aioredis.Redis):
    while True:
        try:
            async with async_timeout.timeout(2):
                message = await channel.get_message(ignore_subscribe_messages=True)
                if message is not None:
                    for key in message:
                        if (isinstance(message[key], bytes)):
                            message[key] = message[key].decode()

                    print(f"(Reader) Message Received: {message}")
                    if message["data"] == STOPWORD:
                        print("(Reader) STOP")
                        break
                    # await redis.publish("sensor", "ok")
                await asyncio.sleep(0.01)
        except asyncio.TimeoutError:
            pass
        except CancelledError:
            pass

async def main():
    redis = aioredis.from_url("redis://localhost")
    pubsub = redis.pubsub()
    await pubsub.psubscribe('sensor.reply')
    asyncio.create_task(reader(pubsub, redis))

asyncio.run(main())