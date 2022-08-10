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