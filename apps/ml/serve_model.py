import tensorflow as tf
from tensorflow.keras import Model
import os
import logging
from typing import Union

from ml.models.internal_self import InternalSelfModel
from ml_common.redis_message import ModelOption
from ml_common.singleton_instance import SingletonInstance
import models

logger = logging.getLogger(__name__)

class ServeModel(SingletonInstance):
    modelPath = "ml_models"
    models = {}

    def __init__(self):
        self.loadModels()

    def loadModels(self):
        modelDirs = os.listdir(self.modelPath)
        for dir in modelDirs:
            try: 
                self.models[dir] = tf.keras.models.load_model(os.path.join(self.modelPath, dir))
            except Exception:
                pass

    def __getModel(self, modelOption: ModelOption) -> Union[InternalSelfModel, Model]:
        return models.create(modelOption.model), self.models[modelOption.model]
    
    def predict(self, modelOption: ModelOption):
        try:
            model, kerasModel = self.__getModel(modelOption)
            pred, input = model.predict(kerasModel, modelOption)
            return pred, input
        except Exception as e:
            logger.exception("error", e)
            return "error"

if __name__ == '__main__':
    serveModel = ServeModel.instance()