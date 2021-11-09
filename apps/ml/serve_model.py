import tensorflow as tf
import os
import logging
from typing import Union
from databases.core import Database

from ml_common import DbConn, Config, ModelOption, SingletonInstance
from models import InternalSelfModel, InferModel
from models import create as modelCreate

logger = logging.getLogger(__name__)

class ServeModel(SingletonInstance):
    modelPath = "ml_models"
    models = {}
    dbConn: Database = None

    def __init__(self):
        self.loadModels()

    def loadModels(self):
        modelDirs = os.listdir(self.modelPath)
        for dir in modelDirs:
            try: 
                kerasModel = tf.keras.models.load_model(os.path.join(self.modelPath, dir))
                model = modelCreate(dir, kerasModel)
                self.models[dir] = model
            except Exception as e:
                logger.exception(e)

    def __getModel(self, modelOption: ModelOption) -> Union[InferModel, InternalSelfModel]:
        return self.models[modelOption.model]
    
    async def predict(self, modelOption: ModelOption):
        try:
            model = self.__getModel(modelOption)
            if (self.dbConn is None or not self.dbConn.is_connected):
                self.dbConn = await DbConn().connect(Config.DB_TYPE, Config.DB_HOST, Config.DB_PORT, Config.DB_DATABASE, Config.DB_USER, Config.DB_PASSWORD)
            pred, input = await model.predictFromDb(self.dbConn, modelOption)
            return pred, input
        except Exception as e:
            logger.exception("error", e)
            return "error"

if __name__ == '__main__':
    serveModel = ServeModel.instance()