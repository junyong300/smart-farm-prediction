# asyncio 를 사용하고 있는데, 현재 한 프로세스에 하나의 train만 하고 있기 때문에 별 의미는 없다
# 하지만 향후 서비스 형태로 train을 할 경우 필요할 수 있기 때문에 databases 패키지와 asyncio를 활용한다

import asyncio
import logging
import sys
import logging_setup
import models
from model_option import ModelOption
from config import Config

conf = Config()
conf.load()

# setup("ai-train.log", logging.DEBUG)
logging_setup.setup("ai-train.log", conf.LOG_LEVEL)
logger = logging.getLogger(__name__)

# 한 프로세스에 하나의 train만 진행한다
async def main(modelId):
    logger.info("AI Train Start!")

    option = ModelOption(modelId, conf)
    model = models.create(option)
    if model:
        ds = await model.makeDataset()
        #if hasattr(option, 'test') and option.test:
        if option.get('test'):
            model.loadModel()
            model.test(ds)
        else:
            model.train(ds)

asyncio.run(main(sys.argv[1]))