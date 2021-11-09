from abc import *
import logging
import tensorflow as tf
from tensorflow.keras import Model
from ml_common.redis_message import ModelOption

class BaseModel(object, metaclass=ABCMeta):
    logger = logging.getLogger(__name__)

    model: Model

    @abstractmethod
    async def fetchDb(self, dbConn, modelOption: ModelOption):
        pass
