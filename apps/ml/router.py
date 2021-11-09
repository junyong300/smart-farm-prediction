import json
from ml_common.redis_message import RedisMessage
from serve_model import ServeModel

async def handler(redis, msg):
    redisMsg = RedisMessage(msg)

    pred, input = await ServeModel.instance().predict(redisMsg.modelOption)
    data = {'input': input, 'pred': pred}

    res = json.dumps({'id': redisMsg.id, 'isDisposed': True, 'response': data})
    await redis.publish("ml.serving.reply", res)
