import asyncio
from asyncio.exceptions import CancelledError
import async_timeout
import aioredis
import logging
import os, sys, pathlib
from signal import SIGINT, SIGTERM, Signals
from ml_common.logger import setup
import router

print(__file__)

logger = logging.getLogger(__name__)

async def reader(channel: aioredis.client.PubSub, redis: aioredis.Redis):
    while True:
        try:
            async with async_timeout.timeout(2):
                msg = await channel.get_message(ignore_subscribe_messages=True)
                
                if msg is not None:
                    for key in msg:
                        if (isinstance(msg[key], bytes)):
                            msg[key] = msg[key].decode()

                    logger.debug(f"(Reader) Message Received: {msg}")
                    router.route(msg)

                    '''
                    if msg["data"] == STOPWORD:
                        print("(Reader) STOP")
                        break
                    '''
                    # await redis.publish("sensor", "ok")
                await asyncio.sleep(0.01)
        except asyncio.TimeoutError:
            pass
        except CancelledError as e:
            break
        except Exception as e:
            logger.exception("Uncatched Error!")

def interrupted(signal: Signals, task: asyncio.Task):
    task.cancel()
    logger.info("Interrupted by " + signal.name)

async def main():
    setup("ml.log", logging.DEBUG)
    logger.info("ML Start!")
    redis = aioredis.from_url("redis://localhost", decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.psubscribe('ml.train')
    await pubsub.psubscribe('ml.predict')
    #await pubsub.psubscribe('ml.reply')

    loop = asyncio.get_event_loop()
    task = asyncio.create_task(reader(pubsub, redis))

    for signal in [SIGINT, SIGTERM]:
            loop.add_signal_handler(signal, interrupted, signal, task)
    
asyncio.run(main())