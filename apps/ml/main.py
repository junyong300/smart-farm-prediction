import asyncio, aioredis
from asyncio.exceptions import CancelledError
from signal import SIGINT, SIGTERM, Signals
import logging

from ml_common.logger import setup
import router

logger = logging.getLogger(__name__)

async def reader(channel: aioredis.client.PubSub, redis: aioredis.Redis):
    while True:
        try:
            msg = await channel.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if msg is not None:
                logger.debug(f"(Reader) Message Received: {msg}")
                # TODO: multiprocessing
                asyncio.create_task(router.route(redis, msg))
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
    setup("mlserving.log", logging.DEBUG)
    logger.info("ML Serving Start!")
    redis = aioredis.from_url("redis://localhost", decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe('ml.serving')

    loop = asyncio.get_event_loop()
    task = asyncio.create_task(reader(pubsub, redis))

    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, interrupted, signal, task)
    
    await task
    
asyncio.run(main())