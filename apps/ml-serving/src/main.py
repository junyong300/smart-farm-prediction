import asyncio
from asyncio.exceptions import CancelledError
import async_timeout
import aioredis

STOPWORD = "STOP"

async def reader(channel: aioredis.client.PubSub, redis: aioredis.Redis):
    while True:
        try:
            async with async_timeout.timeout(2):
                message = await channel.get_message(ignore_subscribe_messages=True)
                if message is not None:
                    print(f"(Reader) Message Received: {message}")
                    if message["data"].decode('utf-8') == STOPWORD:
                        print("(Reader) STOP")
                        break
                    # await redis.publish("sensor", "ok")
                await asyncio.sleep(0.01)
        except asyncio.TimeoutError:
            pass
        except CancelledError:
            pass

async def main():
    print("start")

    redis = aioredis.from_url("redis://localhost")
    pubsub = redis.pubsub()
    # await pubsub.psubscribe("channel:*")
    #await pubsub.psubscribe('{ "cmd": "sensor" }_ack')
    await pubsub.psubscribe('sensor.reply')
    # await redis.set("py-key", "py-value")
    #value = await redis.get("py-key")
    asyncio.create_task(reader(pubsub, redis))

    print("listening")

 #   await redis.publish("channel:1", "Hello")
 #   await redis.publish("channel:2", "World")
 #   await redis.publish("channel:1", STOPWORD)

    print("done")

asyncio.run(main())