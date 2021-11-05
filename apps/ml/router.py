import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import asyncio
import json
from ml_common.redis_message import RedisMessage
from serve_model import ServeModel
from time import sleep

async def route(redis, msg):
    redisMsg = RedisMessage(msg)

    pred, input = ServeModel.instance().predict(redisMsg.modelOption)
    data = {'input': input, 'pred': pred}

    res = json.dumps({'id': redisMsg.id, 'isDisposed': True, 'response': data})
    await redis.publish("ml.serving.reply", res)
    return 'ok'
    #return res
