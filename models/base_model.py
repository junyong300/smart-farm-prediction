from abc import *
import datetime
import tensorflow as tf
import keras
from tensorflow.python.data.ops.dataset_ops import Dataset
import pandas as pd
import os
import logging
from dbconn import DbConn
from config import Config
from model_option import ModelOption
from plot import saveFig

class BaseModel(object, metaclass=ABCMeta):
    logger = logging.getLogger(__name__)
    model: keras.Model

    def __init__(self, option: ModelOption):
        self.option = option
        self.logger.info(option)

        log_dir = "logs/fit/" + self.name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.strategy = tf.distribute.MirroredStrategy()
        with self.strategy.scope():
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def __str__(self):
        return str(self.option)

    @property
    def name(self):
        return self.option.model

    @property
    def id(self):
        return self.option.modelId
    
    async def getConn(self):
        conf = self.option.conf;
        conn = await DbConn().connect(conf.DB_TYPE_LEGACY, conf.DB_HOST_LEGACY, conf.DB_PORT_LEGACY, conf.DB_DATABASE_LEGACY,
                conf.DB_USER_LEGACY, conf.DB_PASSWORD_LEGACY.strip('"'))
        return conn
    
    async def makeDataset(self):
        rs = await self.loadOrFetch()
        seqs = self.preprocess(rs)
        self.logger.info("inter model preprocessing done")

        input_seqs, label_seqs = zip(*seqs)
        input_tensor = tf.constant(input_seqs)
        label_tensor = tf.constant(label_seqs)
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor, label_tensor))

        return dataset

    async def loadOrFetch(self):
        """
        feather 파일이 있으면 사용하고, 없으면 db에서 불러 와서 feather 파일로 저장
        """
        fileName = os.path.join("temp", self.id + ".feather")
        if os.path.isfile(fileName):
            df = pd.read_feather(fileName)
        else:
            rs = await self.fetchDb()
            df = pd.DataFrame(rs)
            df.columns = rs[0].keys()
            if not os.path.exists("temp"):
                os.makedirs("temp")
            df.to_feather(fileName)

        self.logger.info("Model recordset loaded")

        return df

    @abstractmethod
    async def fetchDb(self):
        pass

    @abstractmethod
    def preprocess(self, rs):
        pass

    def train(self, dataset: Dataset):
        self.logger.info("Train start")

        split_point = int(len(dataset) * 0.90)
        trainset = dataset.take(split_point)
        # sequence들이 10분 갭으로 만들어져 있어서 trainset과 testset이 계속 겹치기 때문에 대략 이틀치(5 devices)를 띄운다
        testset = dataset.skip(split_point + 1000)
        self.logger.info(F"Train Size: {len(trainset)}, Test Size: {len(testset)}")

        trainset = trainset.shuffle(buffer_size=100000, reshuffle_each_iteration=True).batch(256)
        testset = testset.batch(256)

        self.model.fit(trainset, epochs=self.option.get('epoch', 50), validation_data=testset, callbacks=[self.tensorboard_callback])

        #self.model.save(self.saved_model_path)
        self.saveModel()
        self.logger.info("Model saved")

    def test(self, dataset):
        pass

    def predict(self, dataset):
        return self.model.predict(dataset)

    def saveModel(self):
        self.model.save(os.path.join("ai_models", self.name))

    def loadModel(self):
        self.model = keras.models.load_model(os.path.join("ai_models", self.name))
