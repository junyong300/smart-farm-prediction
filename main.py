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
def main(modelId):
    logger.info("AI Train Start!")

    option = ModelOption(modelId, conf)
    model = models.create(option)
    if model:
        #if hasattr(option, 'test') and option.test:
        mode = option.get('mode')
        if mode == 'test':
            model.loadModel()
            model.test()
        elif mode == 'poc':
            model.self_proof()
        else:
            ds = model.makeDataset()
            model.train(ds)

main(sys.argv[1])
