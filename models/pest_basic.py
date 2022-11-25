import asyncio
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.python.data import Dataset
#import tensorflow_decision_forests as tfdf
import datetime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob
import json

from plot import saveFig
from dbconn import DbConn
from model_option import ModelOption
import normalize as norm
from .base_model import BaseModel
from .pest.pest_parser import load_csv 

class PestBasicModel(BaseModel):
    #farm_names = [
    #    "김세광", "김창호", "김철곤", "박경종", "손미자", "양이원", "양회경", "",
    #    "김성초", "노홍기", "문병도", "문숙주", "문순애", "문주형", "양동", "양조승", "오병용", "윤영관",
    #            "이병권", "이석철", "이승두", "이시성", "전광희", "차주성", "하동현", "한상용", "황인성"]
    #farm_names = ["김성초", "노홍기", "문병도", "문숙주", "문순애"]
    devices = {}
    pests = [8, 14, 39, 10, 31] # 잎곰팡이, 흰가루병, 시들음병, 잿빛곰팡이, TYLCV
    days = 14

    def __init__(self, option: ModelOption):
        super().__init__(option)

        with self.strategy.scope():
            env_input_width = 24 * self.days
            env_input_shape = (env_input_width, 6)
            crop_input_shape = (1,)
            label_width = len(self.pests) + 1
            label_shape = (label_width,)

            env_input_layer = tf.keras.layers.Input(env_input_shape)
            crop_input_layer = tf.keras.layers.Input(crop_input_shape)

            lstm_layer = tf.keras.layers.LSTM(512, dropout=0.3, input_shape=env_input_shape)
            env_dense_layer = tf.keras.layers.Dense(256)
            crop_dense_layer = tf.keras.layers.Dense(4)

            concat_layer = tf.keras.layers.concatenate([env_dense_layer(lstm_layer(env_input_layer)), crop_dense_layer(crop_input_layer)])
            dense_layer1 = tf.keras.layers.Dense(64)
            dense_layer2 = tf.keras.layers.Dense(32, activation='relu')
            dense_layer3 = tf.keras.layers.Dense(label_width, activation='softmax')
            #reshape_layer = tf.keras.layers.Reshape(label_shape)

            #output_layer = reshape_layer(dense_layer2(dense_layer1(concat_layer)))
            #output_layer = reshape_layer(dense_layer2(env_input_layer))
            output_layer = dense_layer3(dense_layer2(dense_layer1(concat_layer)))
            #output_layer = tf.keras.layers.Softmax()(dense_layer3(dense_layer2(dense_layer1(concat_layer))))

            self.model = tf.keras.Model(inputs=[env_input_layer, crop_input_layer], outputs=output_layer)
            self.model.compile(
                #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                optimizer=tf.optimizers.Adam(learning_rate=0.0005, weight_decay=0.0001),
                metrics=[tf.metrics.BinaryAccuracy(), tf.metrics.TruePositives(), tf.metrics.TrueNegatives(), tf.metrics.FalsePositives(), tf.metrics.FalsePositives()]
            )
        #self.model =  tfdf.keras.RandomForestModel()
        #self.model.compile(metrics=["accuracy"])

        self.model.summary()

    def load_pest(self):
        rs = load_csv()
        return rs

    async def load_devices(self):
        conn = await DbConn().connect("mysql", "ep.farmcloud.kr", 3306, "ep_farm_db", "dev", ".fc12#$")

        target_devices = ','.join(map(str, self.option.devices))
        sql = ("select u.USER_NAME name, IN_ENV_DEVICE_IDX idx from farm_dong fd, user_info u, user_farm uf "
            "where u.USER_IDX = uf.USER_IDX "
            "and fd.FARM_IDX = uf.FARM_IDX "
            "and EXT_WEATHER_DEVICE_IDX is not null"
        )
        self.logger.debug("sql:" + sql)
        rs = await conn.fetch_all(sql)
        await conn.disconnect()
        self.devices = {}
        for r in rs:
            name = str.split(r[0], " ")[0]
            self.devices[name] = r[1]
        
        print(self.devices.values())

    async def fetchEnv(self):
        conn = await DbConn().connect("mysql", "ep.farmcloud.kr", 3306, "ep_farm_db", "dev", ".fc12#$")

        #target_devices = []
        #for farm in self.farm_names:
        #    idx = self.devices[farm]
        #    if idx != None:
        #        target_devices.append(idx)

        #target_devices = ','.join(map(str, self.option.devices))
        #target_devices = ','.join(map(str, target_devices))
        sql = ("select device_idx didx, sensing_dt - INTERVAL EXTRACT(SECOND FROM sensing_dt) SECOND sdt, sie_temp t, sie_humidity h from sdh_internal "
                #F"where device_idx in ({target_devices}) "
                #"and daykey between 20210301 and 20211231 "
                #"and daykey between 20210301 and 20210901 "
                #"where daykey between 20221020 and 20221101 "
                "where sie_temp between -50 and 80 "
                "and sie_humidity between 0 and 100 "
                #"and sie_co2 between 0 and 5000 "
                "order by didx, sdt"
        )
        self.logger.debug("sql:" + sql)
        rs = await conn.fetch_all(sql)
        await conn.disconnect()
        return rs

    def preprocess(self, rs):
        rs_e = rs[0]
        df:DataFrame = rs[1]
        seqs_e = self.preprocess_env(rs_e)
        grs = df.groupby(['farm'])

        seqs_e1h = []
        seqs_pest = []
        seqs_crop = []
        seqs_farm = []

        for gr in grs:
            print(gr[0])
            grdf = gr[1]
            didx = self.devices[gr[0]]
            print(grdf.index.values)
            is_tomato = 1 if grdf.index.values[0][1] == '방울토마토' else 0
            #date = gr[0][2]
            cols = grdf.columns

            try:
                seqs_fe = seqs_e[didx]
            except:
                continue

            eidx = 0 # 환경 index

            for i, r in enumerate(grdf.values):
                # 현재 측정일의 이전 2주일을 input 데이터로 사용한다
                date = datetime.datetime.strptime(str(grdf.index.values[i][2]), '%y%m%d')
                pts = date.timestamp()
                pts_start = pts - 1440 * 60 * self.days 
                pts_end = pts - 60 * 60 # 1시간 전
                start_date = datetime.datetime.fromtimestamp(pts_start)
                #print(start_date, date)

                ets_start = None
                ets_end = None

                # 환경데이터가 측정일 이전 days보다 이전이면 계속 skip한다
                while True:
                    eidx_end = eidx + 60 * 24 * self.days - 60
                    if eidx_end >= len(seqs_fe):
                        #print("eidx_end >= seq_fe len", eidx)
                        break
                    ets_start = seqs_fe[eidx][0]
                    ets_end = seqs_fe[eidx_end][0]
                    if ets_start >= pts_start:
                        #print("ets_start > pts_start", eidx)
                        break
                    eidx = eidx + 1

                if (ets_start == None or eidx_end >= len(seqs_fe)):
                    #print("ets_start is none or eidx_end > seqs_fe len", eidx)
                    break

                #측정일이 2주 전 환경보다 앞이면 건너뛴다
                if pts < ets_start:
                    #print("pts < ets_start")
                    continue

                if (ets_start == pts_start and ets_end == pts_end):
                    print(start_date, date, datetime.datetime.fromtimestamp(ets_start), datetime.datetime.fromtimestamp(ets_end))
                    seq_e1h = seqs_fe[eidx:eidx_end + 60:60]
                    seqs_e1h.append(seq_e1h)

                    pests = [1 if r[p] > 0 else 0 for p in self.pests]
                    normal = 1 if all(e == 0 for e in pests ) else 0
                    pests = [normal] + pests
                    seqs_pest.append(pests)
                    seqs_crop.append(is_tomato)
                    seqs_farm.append([gr[0], grdf.index.values[0][1], seq_e1h[0]])
                #print(pests)
        return seqs_e1h, seqs_pest, seqs_crop, seqs_farm


    def preprocess_env(self, rs):
        min_point = 60 * 24 * 7
        seq = []
        seqs = {}

        if isinstance(rs, DataFrame):
            rs = rs.to_dict('records')

        for i in tqdm(range(len(rs) - 1)):
            cidx = rs[i]['didx']
            csdt = rs[i]['sdt']
            nsdt = rs[i + 1]['sdt']
            interval = int((nsdt - csdt).total_seconds() / 60) # 분

            if cidx != rs[i+1]['didx']: # device가 변경되면 append하고 새 seq를 시작
                if (len(seq) > min_point):
                    seqs[cidx] = seq
                cidx = rs[i+1]['didx']
                seq = []
            else:
                seq_t = self.fill_missing(rs[i], rs[i+1], csdt, interval)
                seq.extend(seq_t)

        return seqs
    
    def fill_missing(self, row_prev, row_next, csdt, interval):
        '''
        1보다 긴 interval인 경우 interval이 1이 되도록 빈 시간을 채운다
        interval이 60을 채우지 않고 현재 값만 넣는다
        '''
        cts = datetime.datetime.timestamp(csdt)
        seq = []

        if interval > 60:
            interval = 1

        for i in range(interval):
            ts = cts + i * 60

            tr = row_prev['t'] + (row_next['t'] - row_prev['t']) * i / interval
            hr = row_prev['h'] + (row_next['h'] - row_prev['h']) * i / interval
            #co2r = row_prev['co2'] + (row_next['co2'] - row_prev['co2']) * i / interval

            # normalize
            t = norm.t_norm(tr)
            h = hr / 100
            #l = norm.l_norm(lr)
            #co2 = norm.co2_norm(co2r)
            tsn = list(norm.timestamp_to_sincos(ts))

            #rt = [tr, hr, co2r, ts, t, h, co2] + tsn
            rt = [ts, t, h] + tsn
            seq.append(rt)

        return seq
    

    def train(self, dataset: Dataset):
        self.logger.info("Train start")

        split_point = int(len(dataset) * 0.90)
        dataset = dataset.shuffle(2000)
        trainset = dataset.take(split_point)
        #trainset = dataset
        # sequence들이 10분 갭으로 만들어져 있어서 trainset과 testset이 계속 겹치기 때문에 대략 이틀치(5 devices)를 띄운다
        testset = dataset.skip(int(split_point * 0.7))
        #testset = dataset.skip(split_point)
        self.logger.info(F"Train Size: {len(trainset)}, Test Size: {len(testset)}")

        #trainset = trainset.shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(8)
        trainset = trainset.batch(16)
        testset = testset.batch(16)


        #self.model.fit(trainset, epochs=self.option.get('epoch', 50), validation_data=testset, callbacks=[self.tensorboard_callback])
        self.model.fit(trainset, validation_data=testset, callbacks=[self.tensorboard_callback], epochs=500)

        #self.model.save(self.saved_model_path)
        self.saveModel()
        self.logger.info("Model saved")

    def evaluate(self, testset, farm):
        result = self.model.evaluate(testset)
        result2 = dict(zip(self.model.metrics_names, result))
        print(result2)

        pred = self.model.predict(testset)
        print(pred)
        head = tf.estimator.MultiLabelHead(6)



        inputs_ds = testset.map(lambda i, l: i[0]).unbatch()
        inputs = np.stack(list(inputs_ds))
        inputs = norm.t_denorm(inputs)

        labels_ds = testset.map(lambda i, l: l).unbatch()
        labels = np.stack(list(labels_ds))
        features = {'x': inputs}
        loss = head.loss(labels, pred, features={'x': inputs})
        print(loss.numpy())
        eval_metrics = head.metrics()
        updated_metrics = head.update_metrics(
        eval_metrics, features, pred, labels)
        for k in sorted(updated_metrics):
            print('{} : {:.2f}'.format(k, updated_metrics[k].result().numpy()))

        self.plot_value_array(pred, labels, farm)
    
    def plot_value_array(self, preds, labels, farm):
        plt.rcParams["font.family"] = 'NanumGothicCoding'
        i = 0

        rows = 8 
        cols = 6

        plt.figure(figsize=(2*cols, 2*rows))
        for i in range(rows * cols):
            plt.subplot(rows, cols, i+1)
            predictions_array = preds[i]
            true_label = labels[i]
            plt.grid(False)
            range(6)
            #plt.xticks(range(6))
            name = farm['0'][i]
            crop = farm['1'][0]
            date = json.loads(farm['2'][0])[0]
            date = datetime.datetime.fromtimestamp(date).strftime("%Y%m%d")
            title = name[0] + "*" + (name[2] if len(name) > 2 else "") + " - " + crop + " " + date
            plt.title(title)
            plt.xticks(range(6), labels=["없음", "잎곰팡이", "흰가루병", "시들음병", "잿빛곰팡이", "TYLCVi"], rotation=45)
            plt.yticks([])
            thisplot = plt.bar(range(6), predictions_array, color="#777777")
            plt.ylim([0, 1])
            predicted_label = np.argmax(predictions_array)
            thisplot[predicted_label].set_color('red')
            for i, t in enumerate(true_label):
                if t > 0:
                    thisplot[i].set_color('blue')

        plt.tight_layout()
        plt.show()
        prefix = "pest"
        if not os.path.exists(F"./ai_models/fig/{prefix}"):
            os.makedirs(F"./ai_models/fig/{prefix}")

        for f in glob.glob(F"./ai_models/fig/{prefix}/{prefix}*.jpg"):
            os.remove(f)
        plt.savefig(F"./ai_models/fig/{prefix}/{prefix}.jpg")

'''
async def main():
    option = ModelOption("pest", None)
    model = PestModel(option)

    data_path = "temp/pest/tfdata"

    try:
        dataset = tf.data.experimental.load(data_path)
    except:
        await model.load_devices()
        rs_e = await model.fetchEnv()
        rs_p = model.load_pest()

        env_seqs, pest_seqs, crop_seqs, farm_seqs = model.preprocess((rs_e, rs_p))

        df = pd.DataFrame(farm_seqs)
        df.to_csv("temp/pest/pest_env.csv", encoding='utf-8-sig')

        env_seqs = np.array(env_seqs)
        env_seqs = np.delete(env_seqs, [0], axis=2)

        env_tensor = tf.constant(env_seqs)
        crop_tensor = tf.constant(crop_seqs)
        pest_tensor = tf.constant(pest_seqs)

        dataset = tf.data.Dataset.from_tensor_slices(((env_tensor, crop_tensor), pest_tensor))
        tf.data.experimental.save(dataset, "temp/pest/tfdata")
    
    
    df = pd.read_csv('temp/pest/pest_env.csv')
    testset = dataset.skip(int(len(dataset) * 0.8))

    model.train(dataset)
    #model.loadModel()
    model.evaluate(dataset.batch(32), df.to_dict())


if __name__ == '__main__':
    asyncio.run(main())
'''