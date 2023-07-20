import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
import datetime
import os
from plot import saveFig

from dbconn import DbConn
from model_option import ModelOption
import normalize as norm
from .base_model import BaseModel
from .aihub.aihub_poc import load_aihub_envs

class InternalPocModel(BaseModel):
    ''' 내부 24시간 + 기상청 데이터로 내부 온습도 예측 '''
    def __init__(self, option: ModelOption):
        super().__init__(option)

        multi = [1, 0.1]

        def loss_weighted(y_true, y_pred):
            err = (y_true - y_pred) * multi
            score = tf.math.reduce_mean(tf.math.square(err))
            return score


        def t_mae_o(y_true, y_pred):
            ae = abs(y_pred[:,:,0] - y_true[:,:,0])
            score = np.average(ae)

            return score
        
        def t_mae(y_true, y_pred):
            score = tf.py_function(func=t_mae_o, inp=[y_true, y_pred], Tout=tf.float32,  name='t_mae') 
            return score

        with self.strategy.scope():
            env_input_width = 60 * 24 * 2 // 10 
            env_input_shape = (env_input_width, 6)
            wfs_input_width = 42 
            wfs_input_shape = (wfs_input_width, 5)
            label_width = 60 * 24 * 2 // 10
            label_shape = (label_width, 2)

            env_input_layer = tf.keras.layers.Input(env_input_shape)
            wfs_input_layer = tf.keras.layers.Input(wfs_input_shape)

            lstm_layer = tf.keras.layers.LSTM(288, dropout=0.1, input_shape=env_input_shape)
            env_dense_layer = tf.keras.layers.Dense(288 * 2)

            wfs_dense_layer = tf.keras.layers.Dense(64)
            wfs_out_layer = wfs_dense_layer(wfs_input_layer)
            wfs_out_layer = tf.keras.layers.Flatten()(wfs_out_layer)

            concat_layer = tf.keras.layers.concatenate([env_dense_layer(lstm_layer(env_input_layer)), wfs_out_layer])
            dense_layer1 = tf.keras.layers.Dense(512)
            dense_layer2 = tf.keras.layers.Dense(label_width * 2)
            reshape_layer = tf.keras.layers.Reshape(label_shape)
            output_layer = reshape_layer(dense_layer2(dense_layer1(concat_layer)))

            self.model = tf.keras.Model(inputs=[env_input_layer, wfs_input_layer], outputs=output_layer)

            self.model.compile(
                #loss=tf.losses.MeanSquaredError(),
                loss=loss_weighted,
                optimizer=tf.optimizers.Adam(learning_rate=0.002),
                metrics=[tf.metrics.MeanAbsoluteError(), t_mae]
            )

        self.model.summary()

    async def makeDataset(self) -> Dataset:
        rs = await self.loadOrFetch()
        env_seqs, wfs_seqs = self.preprocess(rs)
        self.logger.info("preprocessing done")

        # make dataset
        input_seqs, label_seqs = zip(*env_seqs)
        input_seqs = np.array(input_seqs)
        input_seqs = np.delete(input_seqs, [0], axis=2) # sdt 삭제

        #for csv
        df_in = pd.DataFrame(input_seqs.tolist())
        df_wfs = pd.DataFrame(wfs_seqs)
        df_label = pd.DataFrame(label_seqs)
        if not os.path.exists("temp/ep"):
            os.makedirs("temp/ep")
        df_in.to_csv("temp/ep/in_prep.csv")
        df_wfs.to_csv("temp/ep/wfs_prep.csv")
        df_label.to_csv("temp/ep/label_prep.csv")

        env_input_tensor = tf.constant(input_seqs)
        label_tensor = tf.constant(label_seqs)
        wfs_input_tensor = tf.constant(wfs_seqs)

        dataset = tf.data.Dataset.from_tensor_slices(
            ((env_input_tensor, wfs_input_tensor), label_tensor))
        return dataset

    async def loadOrFetch(self):
        """ external과 wfs 각각에 대해 처리한다 """

        # internal
        nx = 48
        ny = 111
        df_in = load_aihub_envs()
        df_in.columns = ['sdt', 't', 'h', 'didx']
        df_in = df_in.sort_values(by=['didx', 'sdt']) # 이미 정렬되어 있어서 필요없지만 혹시나 놔둔다
        df_in = df_in.dropna()
        df_in = df_in.reset_index()
        fileName = os.path.join("temp", self.id + "_aihub.feather")
        df_in.to_feather(fileName)

        nx = 58 # 126.852 aihub의 농장 정보는 없지만 광주로 간주하여 처리한다
        ny = 74 # 35.159 
        # wfs
        # feather 파일이 있으면 사용하고, 없으면 db에서 불러 와서 feather 파일로 저장
        fileName = os.path.join("temp", self.id + "_wfs.feather")
        if os.path.isfile(fileName):
            df_wfs = pd.read_feather(fileName)
        else:
            rs = await self.fetchWfs(nx, ny)
            df_wfs = pd.DataFrame(rs)
            df_wfs.columns = rs[0].keys()
            if not os.path.exists("temp"):
                os.makedirs("temp")
            
            df_wfs.to_feather(fileName)

        wfs_start_dt = df_wfs.iloc[0, 0]
        wfs_end_dt = df_wfs.iloc[len(df_wfs) - 1, 0]

        df_in = df_in.loc[((df_in.sdt + datetime.timedelta(days=4)) >= wfs_start_dt)
                            & (df_in.sdt - datetime.timedelta(days=2) <= wfs_end_dt), :]

        print(len(df_in), len(df_wfs))
        self.logger.info("Model recordset loaded")

        return (df_in, df_wfs)
    
    async def fetchWfs(self, nx, ny):
        conn = await DbConn().connect("mysql", "farmcloud.kr", 3307, "collect", "root", ".fc12#$")
        sql = (
            "SELECT base_dt bdt, fcst_dt fdt, tmp "
            "FROM kma_villagefcst_gen "
            F"WHERE nx = {nx} and ny = {ny} "
            "AND tmp between -50 and 50 "
            "HAVING TIMESTAMPDIFF(HOUR, base_dt, fcst_dt) <= 48 + 3 "   # 48 + 3시간
            "ORDER BY base_dt, fcst_dt "
            # "LIMIT 10000 "
        )

        rs = await conn.fetch_all(sql)
        await conn.disconnect()
        return rs
    
    def preprocess(self, rs):
        env_seqs_all = self.preprocess_env(rs[0])
        df_wfs = rs[1]

        env_seqs = []
        env_input_width = 60 * 24 * 2
        gr_wfs = df_wfs.groupby(['bdt'])
        wfs_seqs = []
        wfs_idx = 0
        grs = [group for bdt, group in gr_wfs]
        bts = 0
        lastSts = 0

        for i, seq in enumerate(env_seqs_all):
            sts = seq[0][env_input_width // 10 - 1][0] # sensing dt
            if lastSts > sts or wfs_idx >= len(grs): # env_seqs_all 에서 device가 바뀌면 다시 과거 날짜로 돌아가므로 wfs도 처음으로 되돌린다
                wfs_idx = 0
            bts = datetime.datetime.timestamp(grs[wfs_idx]['bdt'].iloc[0]) # base dt

            # bdt가 sts보다 3시간 이전이면 아니게 될 때까지 wfs 건너 뜀
            # while len(grs[wfs_idx]) != 16 or (bts + 3600 * 3 < sts + 600) :
            while (bts + 3600 * 3 < sts + 600) :
                wfs_idx += 1
                if wfs_idx >= len(grs):
                    break
                bts = datetime.datetime.timestamp(grs[wfs_idx]['bdt'].iloc[0])
            
            # print(datetime.datetime.fromtimestamp(sts), datetime.datetime.fromtimestamp(bts))

            # 필요한 예보인 경우
            if bts <= sts and (bts + 3600 * 3) > sts:
                wts_seq = [[norm.t_norm(float(tmp))] + [item for item in norm.datetime_to_sincos(fdt)] # pop의 비중을 절반으로 줄임
                           for tmp, fdt in grs[wfs_idx][['tmp', 'fdt']].values]
                if (len(wts_seq) >= 42):
                    wfs_seqs.append(wts_seq[:42]) # 48개만 input 데이터로 사용한다
                    env_seqs.append(env_seqs_all[i])
                else:
                    print('wts_seq len: ', len(wts_seq))
            
            lastSts = sts

        return env_seqs, wfs_seqs

    def preprocess_env(self, rs):
        input_width = 60 * 24 * 2
        label_width = 60 * 24 * 2
        min_point = input_width + label_width
        device_data_count = 0
        seq = []
        seqs = []

        if isinstance(rs, DataFrame):
            rs = rs.to_dict('records')

        for i in range(len(rs) - 1):
            cdidx = rs[i]['didx']
            csdt = rs[i]['sdt'].replace(second=0, microsecond=0) # current
            nsdt = rs[i + 1]['sdt'].replace(second=0, microsecond=0) # next
            interval = int((nsdt - csdt).total_seconds() / 60) # 분

            if interval > 30 or cdidx != rs[i+1]['didx']:
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
                if cdidx != rs[i+1]['didx']:
                    cdidx = rs[i+1]['didx']
                    device_data_count = 0

        refined_seqs = []

        for seq in seqs:
            for i in range(0, len(seq) - (input_width + label_width), 11):
                input_seq = seq[i: i + input_width: 10]
                label_seq = [[x[1], x[2]] for x in seq[i + input_width:i + input_width + label_width: 10]] # t, h
                if len(label_seq) == label_width // 10:
                    refined_seqs.append((input_seq, label_seq))

        # wfs 데이터와 sync를 맞출 때 sorting이 되어 있으면 빠르게 처리할 수 있다
        refined_seqs = sorted(refined_seqs, key=lambda seq: seq[0][0][0])

        return refined_seqs

    def fill_missing(self, row_prev, row_next, csdt, interval):
        '''
        1보다 긴 interval인 경우 interval이 1이 되도록 빈 시간을 채운다
        '''
        cts = datetime.datetime.timestamp(csdt)
        seq = []

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