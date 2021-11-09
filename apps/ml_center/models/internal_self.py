import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import tensorflow as tf
from tqdm import tqdm

import datetime
import logging
from databases import Database

from tensorflow.python.data.ops.dataset_ops import Dataset

import models
from .train_model import TrainModel
from ml_common.logger import setup
from ml_common import ModelOption
import ml_common.normalize as norm

logger = logging.getLogger(__name__)

class InternalSelfModel(TrainModel):
    #saved_model_path = './saved_model/sensing'
    '''
        predict 24 hours internal env from 48 hours internal env
    '''
    devices = [115, 62, 79, 91, 116, 63, 61, 51, 19, 103, 78, 60, 54, 39, 47, 70, 80, 99, 21, 119]
    #devices = [115, 62, 79]

    def __init__(self):
        '''
            define tensorflow model
        '''
        super().__init__()
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            input_width = 60 * 24 * 2 // 10 
            #input_shape = (input_width, 7)
            input_shape = (input_width, 6)

            input_layer = tf.keras.layers.Input(input_shape)
            lstm_layer = tf.keras.layers.LSTM(32, dropout=0.1, input_shape=input_shape)
            out_layer = tf.keras.layers.Dense(144)

            self.model = tf.keras.Model(inputs=input_layer, outputs=out_layer(lstm_layer(input_layer)))
            self.model.compile(
                loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=0.0005),
                metrics=[tf.metrics.MeanAbsoluteError()]
            )

            self.model.summary()

    def train(self, dataset: Dataset):
        logger.info("inter model train start")

        split_point = int(len(dataset) * 0.90)
        trainset = dataset.take(split_point)
		# sequence들이 10분 갭으로 만들어져 있어서 trainset과 testset이 계속 겹치기 때문에 대략 이틀치(5 devices)를 띄운다
        testset = dataset.skip(split_point + 1000)
        logger.info(F"Train Size: {len(trainset)}, Test Size: {len(testset)}")

        trainset = trainset.shuffle(buffer_size=100000, reshuffle_each_iteration=True).batch(64)
        testset = testset.batch(64)

        self.model.fit(trainset, epochs=50, validation_data=testset, callbacks=[self.tensorboard_callback])
        #self.model.save(self.saved_model_path)
        self.saveModel()
        logger.info("inter model saved")

    def predict(self, dataset:Dataset):
        return self.model.predict(dataset)

    async def makeDataset(self) -> Dataset:
        rs = await self.loadOrFetch()
        logger.info("inter model recordset loaded")
        seqs = self.preprocess(rs)
        logger.info("inter model preprocessing done")


        input_seqs, label_seqs = zip(*seqs)

        #input_seqs = np.array(input_seqs)
        #assert not np.any(np.isnan(input_seqs))
        #assert not np.any(np.isnan(label_seqs))

        # input_seqs = np.delete(input_seqs, [0, 1, 2, 3], axis=2)
        #input_seqs = np.delete(input_seqs, [0, 1, 2], axis=2)

        input_tensor = tf.constant(input_seqs)
        label_tensor = tf.constant(label_seqs)
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor, label_tensor))

        return dataset

    async def fetchDb(self, dbConn: Database, modelOption: ModelOption ):
        target_devices = ','.join(map(str, self.devices))
        #sql = ("select device_idx didx, sensing_dt sdt, sie_temp t, sie_humidity h, sie_co2 co2 from sdh_internal "
        sql = ("select device_idx didx, sensing_dt sdt, sie_temp t, sie_co2 co2 from sdh_internal "
                F"where device_idx in ({target_devices}) "
                #"and daykey between 20210301 and 20210901 "
                #"and daykey between 20210301 and 20210302 "
                "and sie_temp between -50 and 80 "
                "and sie_co2 between 0 and 5000 "
                "order by didx, sdt"
        )
        logger.debug("sql:" + sql)
        rs = await dbConn.fetch_all(sql)
        return rs
    
    def preprocess(self, rs):
        input_width = 60 * 24 * 2
        label_width = 60 * 24
        min_point = input_width + label_width
        #current_didx = self.devices[0]
        device_data_count = 0
        seq = []
        seqs = []

        if isinstance(rs, DataFrame):
            rs = rs.to_dict('records')

        for i in tqdm(range(len(rs) - 1)):

            current_didx = rs[i]['didx']
            csdt = rs[i]['sdt'].replace(second=0, microsecond=0)
            # next sdt
            nsdt = rs[i + 1]['sdt'].replace(second=0, microsecond=0)
            interval = int((nsdt - csdt).total_seconds() / 60) # 분
        
            if interval > 30 or current_didx != rs[i+1]['didx']:
                # 다음 데이터에서 갭이 발생하거나 device가 변경되는 경우는 지금까지의 sequence를 마무리한다
                # sequence 길이가 min_point(3일치) 이상이면 사용
                if len(seq) >= (min_point - 10):
                    if len(seq) < min_point:  # min_point에서 모자란만큼 채워 넣음
                        seq.extend(self.fill_missing(rs[i], rs[i], csdt, min_point - len(seq)))
                    seqs.append(seq)

                seq = []

            elif interval > 0:
                # 연속됨
                seq.extend(self.fill_missing(rs[i], rs[i+1], csdt, interval))
                device_data_count += 1
                if i == len(rs) - 2:  # 마지막 루프이면 마지막 데이터를 넣는다
                    seq.extend(self.fill_missing(rs[i+1], rs[i+1], nsdt, 1))

                # 다음 포인트에 device 변경되는 경우
                if current_didx != rs[i+1]['didx']:
                    current_didx = rs[i+1]['didx']
                    device_data_count = 0

        refined_seqs = []
        for seq in seqs:
            for i in range(0, len(seq) - (input_width + label_width), 11):
                input_seq = seq[i: i + input_width: 10]
                label_seq = [x[0] for x in seq[i + input_width:i + input_width + label_width: 10]]
                if len(label_seq) == label_width // 10:
                    refined_seqs.append((input_seq, label_seq))
        # refined_seqs = sorted(refined_seqs, key=lambda seq: seq[0][0][3])

        return refined_seqs

    def fill_missing(self, row_prev, row_next, csdt, interval):
        cts = datetime.datetime.timestamp(csdt)
        seq = []

        for i in range(interval):
            ts = cts + i * 60

            tr = row_prev['t'] + (row_next['t'] - row_prev['t']) * i / interval
            #hr = row_prev['h'] + (row_next['h'] - row_prev['h']) * i / interval
            co2r = row_prev['co2'] + (row_next['co2'] - row_prev['co2']) * i / interval

            # normalize
            t = norm.t_norm(tr)
            #h = hr / 100
            #l = norm.l_norm(lr)
            co2 = norm.co2_norm(co2r)
            tsn = list(norm.timestamp_to_sincos(ts))

            #rt = [tr, hr, co2r, ts, t, h, co2] + tsn
            rt = [t, co2] + tsn
            seq.append(rt)

        return seq

