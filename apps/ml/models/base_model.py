from abc import *
import logging
import tensorflow as tf
import pandas as pd
import os
from ml_common.redis_message import ModelOption
from ml_common.dbconn import DbConn

class BaseModel(object, metaclass=ABCMeta):
    logger = logging.getLogger(__name__)
    pgConn = None

    model = tf.keras.Model()

    def __init__(self):
        dbConn = DbConn('postgresql')
        self.pgConn = dbConn.connect("192.168.0.229", 5432, "farmconnect", "postgres", ".fc12#$")

    @abstractmethod
    def fetchDb(self, modelOption: ModelOption):
        pass

    @abstractmethod
    def makeInput(self):
        pass

    @abstractmethod
    def predict(self, model, modelOption: ModelOption):
        pass
