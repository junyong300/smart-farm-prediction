from abc import *
import tensorflow as tf
import pandas as pd
import os
from ml_common.dbconn import DbConn

class BaseModel(object, metaclass=ABCMeta):
    model = tf.keras.Model()

    def __init__(self):
        mysql = DbConn('mysql')
        self.mysqlConn = mysql.connect("192.168.0.221", 3307, "cntd_farm_db", "root", ".fc12#$")
    
    def getName(self):
        return self.__class__.__name__[: -5] # truncate 'Model' from class name

    def loadOrFetch(self):
        fileName = os.path.join("temp", self.getName() + ".feather")
        if os.path.isfile(fileName):
            df = pd.read_feather(fileName)
        else:
            rs = self.fetchTrainDb()
            df = pd.DataFrame(rs)
            if not os.path.exists("temp"):
                os.makedirs("temp")
            df.to_feather(fileName)

        return df

    @abstractmethod
    def fetchTrainDb(self):
        pass

    @abstractmethod
    def makeDataset(self):
        pass

    @abstractmethod
    def train(self, dataset):
        pass

    @abstractmethod
    def predict(self):
        pass

    def saveModel(self):
        self.model.save(os.path.join("ml_models", self.getName()))
