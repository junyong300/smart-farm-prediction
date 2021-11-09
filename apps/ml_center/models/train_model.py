from abc import *
import tensorflow as tf
import pandas as pd
import os
from ml_common import DbConn, BaseModel, Config

class TrainModel(BaseModel, metaclass=ABCMeta):
    model = tf.keras.Model()
    
    def getName(self):
        return self.__class__.__name__[: -5] # truncate 'Model' from class name

    async def loadOrFetch(self):
        fileName = os.path.join("temp", self.getName() + ".feather")
        if os.path.isfile(fileName):
            df = pd.read_feather(fileName)
        else:
            rs = await self.fetchDbWrapper()
            df = pd.DataFrame(rs)
            if not os.path.exists("temp"):
                os.makedirs("temp")
            df.to_feather(fileName)

        return df
    
    
    async def fetchDbWrapper(self):
        conn = await DbConn().connect(Config.DB_TYPE, Config.DB_HOST, Config.DB_PORT, Config.DB_DATABASE, Config.DB_USER, Config.DB_PASSWORD)
        rtn = await self.fetchDb(conn, None)
        await conn.disconnect()

        return rtn

    @abstractmethod
    def makeDataset(self):
        pass

    @abstractmethod
    def train(self, dataset):
        pass

    def saveModel(self):
        self.model.save(os.path.join("ml_models", self.getName()))
