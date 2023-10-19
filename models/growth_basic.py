#import databases
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
import datetime
import os

from dbconn import DbConn

from .base_model import BaseModel
from model_option import ModelOption
import normalize as norm

'''
* input1 env: 1주일 환경(1일 aggregation) d-6 ~ d-0
    daily_farm 주간평균온도, 야간평균온도, 일최고온도, 일최저온도, 습도, co2평균, hd평균
    date: day_sin, day_cos

* input2 growth: 생육 d0
    초장, 줄기룩기, 잎길이, 잎폭
    week 차수

* output: d+7 생육. 초장, 줄기굵기, 잎길이, 잎폭
'''

class GrowthBasicModel(BaseModel):
    def __init__(self, option: ModelOption):
        super().__init__(option)

        with self.strategy.scope():
            env_input_width = 7
            env_input_shape = (env_input_width, 9)
            #grw_input_width = 0
            #grw_input_shape = (grw_input_width, 0)
            grw_input_shape = 5
            label_width = 0
            #label_shape = (label_width, 0)
            label_shape = 4

            env_input_layer = tf.keras.layers.Input(env_input_shape)
            grw_input_layer = tf.keras.layers.Input(grw_input_shape)

            lstm_layer = tf.keras.layers.LSTM(7, dropout=0.1, input_shape=env_input_shape)
            env_dense_layer = tf.keras.layers.Dense(7 * 2)
            grw_dense_layer = tf.keras.layers.Dense(10)

            concat_layer = tf.keras.layers.concatenate([env_dense_layer(lstm_layer(env_input_layer)), grw_dense_layer(grw_input_layer)])

            dense_layer1 = tf.keras.layers.Dense(32)
            dense_layer2 = tf.keras.layers.Dense(4)
            output_layer = dense_layer2(dense_layer1(concat_layer))

            self.model = tf.keras.Model(inputs=[env_input_layer, grw_input_layer], outputs=output_layer)

            self.model.compile(
                loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.metrics.MeanAbsoluteError()]
            )

        self.model.summary()

    async def makeDataset(self) -> Dataset:
        rs = await self.loadOrFetch()
        env_seqs, grw_seqs, label_seqs = self.preprocess(rs)
        self.logger.info("Preprocessing done")

        grw_seqs = np.array(grw_seqs)
        env_seqs = np.array(env_seqs)
        label_seqs = np.array(label_seqs)

        env_seqs = np.delete(env_seqs, [9, 10], axis=2)
        env_seqs = env_seqs.astype(np.float32)

        # nan 을 0으로 변환한다. nan이 포함된 row는 모두 버리는게 맞지만, 현재 데이터가 없어서 임시로 이렇게 처리함.
        grw_seqs = np.nan_to_num(grw_seqs)
        env_seqs = np.nan_to_num(env_seqs)
        label_seqs = np.nan_to_num(label_seqs)

        grw_tensor = tf.constant(grw_seqs)
        env_tensor = tf.constant(env_seqs)
        label_tensor = tf.constant(label_seqs)

        print(grw_tensor.shape, env_tensor.shape, label_tensor.shape)

        dataset = tf.data.Dataset.from_tensor_slices(((env_tensor, grw_tensor), label_tensor))
        return dataset

    async def loadOrFetch(self):
        # growth
        fileName = os.path.join("temp", self.id + "_growth.feather")
        if os.path.isfile(fileName):
            df_growth = pd.read_feather(fileName)
        else:
            rs_growth = await self.fetchGrowth()
            df_growth = pd.DataFrame(rs_growth)
            df_growth.columns = rs_growth[0].keys()
            if not os.path.exists("temp"):
                os.makedirs("temp")
            df_growth.to_feather(fileName);

        farm_idxes = df_growth.farm_idx.unique()

        # env
        fileName = os.path.join("temp", self.id + "_env.feather")
        if os.path.isfile(fileName):
            df_env = pd.read_feather(fileName)
        else:
            rs_env = await self.fetchEnv(farm_idxes)
            df_env = pd.DataFrame(rs_env)
            df_env.columns = rs_env[0].keys()
            if not os.path.exists("temp"):
                os.makedirs("temp")
            df_env.to_feather(fileName);

        return (df_growth, df_env)

    async def getConn(self):
        conn = await DbConn().connect('mysql', "59.3.234.79", 3307, "cntd_farm_db", "root", ".fc12#$")
        return conn

    async def fetchEnv(self, farm_idxes):
        conn = await self.getConn()

        target_farms = ','.join(map(str, farm_idxes))
        #sql = ("select device_idx didx, sensing_dt sdt, sie_temp t, sie_humidity h, sie_co2 co2 from sdh_internal "
        sdate = self.option.sdate.replace("-", "")
        edate = self.option.edate.replace("-", "")
        sql = ("select * from daily_farm "
                #F"where fd_farm_idx in ({target_farms}) "
                # F"and fd_day_key between {sdate} and {edate} "
                #"and temp_day between -50 and 80 "
                "order by fd_farm_idx, fd_day_key"
        )
        self.logger.debug("sql:" + sql)
        rs = await conn.fetch_all(sql)
        return rs

    async def fetchGrowth(self):
        conn = await self.getConn()

        sql = (
            "select fcs_idx, invt_dt, smpl_no, farm_idx, week_idx, init_lnth, stem_thck, leaf_lnth, leaf_wdth from growth_log "
            "order by fcs_idx, smpl_no, invt_dt "
            # "limit 5"
        )

        rs = await conn.fetch_all(sql)
        return rs

    def preprocess(self, _rs):
        env_seqs_all = self.preprocess_env(_rs)

        rs = _rs[0]
        print(rs.head(10))

        input_width = 60 * 24 * 2
        label_width = 60 * 24
        min_point = input_width + label_width
        device_data_count = 0

        seqs_env = []
        seqs_grw = []
        seqs_label = []

        # normalize
        ph_norm = lambda x: (x - 130.3) / 74.7 # plant height
        st_norm = lambda x: (x - 7.7) / 1.9 # stem thick
        ln_norm = lambda x: (x - 18.9) / 12.7 # leaft length
        lw_norm = lambda x: (x - 14.9) / 11.1 # leaft width


        if isinstance(rs, DataFrame):
            rs = rs.to_dict('records')

        for i in range(len(rs) - 1):
            cr = rs[i]
            nr = rs[i+1]

            #다음 row가 새로운 샘플이면 넘어감
            if (cr['smpl_no'] != cr['smpl_no']):
                continue

            curr_date: datetime.date = rs[i]['invt_dt']
            next_date: datetime.date = rs[i+1]['invt_dt']


            days = (next_date - curr_date).days

            # 조사주기가 7일 근처가 아니면 넘어감
            if (days > 9 or days < 5):
                continue


            for j in range(len(env_seqs_all)):
                if env_seqs_all[j][0][10] != cr['farm_idx']:
                    continue

                #print(curr_date, env_seqs_all[j][0][9])
                if (curr_date - env_seqs_all[j][0][9]).days == 0:
                    grw = []
                    grw.append(cr['week_idx'] / 100)
                    grw.append(ph_norm(cr['init_lnth']))
                    grw.append(st_norm(cr['stem_thck']))
                    grw.append(ln_norm(cr['leaf_lnth']))
                    grw.append(lw_norm(cr['leaf_wdth']))

                    ngrw = []
                    ngrw.append(ph_norm(nr['init_lnth']))
                    ngrw.append(st_norm(nr['stem_thck']))
                    ngrw.append(ln_norm(nr['leaf_lnth']))
                    ngrw.append(lw_norm(nr['leaf_wdth']))

                    seqs_grw.append(grw)
                    seqs_env.append(env_seqs_all[j])
                    seqs_label.append(ngrw)
                    break

        return seqs_env, seqs_grw, seqs_label


    def preprocess_env(self, _rs):
        '''
            환경데이터 처리. 이후 preprocess()에서 생육데이터와 sync를 맞춘다
        '''
        rs = _rs[1]
        print(rs.head(10))

        seqs = []
        input_env = []

        if isinstance(rs, DataFrame):
            rs = rs.to_dict('records')

        datetime_format = "%Y%m%d"

        for i in range(len(rs) - 6):
            seq = []
            sr = rs[i]
            nr = rs[i + 1]
            er = rs[i + 6] # 7일을 한 seq로 만들 것이므로, 검증을 위해 마지막날의 날짜가 필요하다

            sfidx = sr['FD_FARM_IDX']
            efidx = er['FD_FARM_IDX']
            #print(sfidx, efidx)

            if sfidx != efidx:
                continue

            start_date = datetime.datetime.strptime(str(sr['FD_DAY_KEY']), datetime_format)
            #next_date = datetime.datetime.strptime(str(nr['FD_DAY_KEY']), datetime_format)
            end_date = datetime.datetime.strptime(str(er['FD_DAY_KEY']), datetime_format)
            ddays = (end_date - start_date).days
            if ddays != 6:
                continue

            for j in range(7):
                k = i + j
                day_env = []
                day_env.append(norm.t_norm(rs[k]['TEMP_AVG_DT'])) #주간온도
                day_env.append(norm.t_norm(rs[k]['TEMP_AVG_NT'])) #야간온도
                day_env.append(norm.t_norm(rs[k]['TEMP_MAX_DAY'])) #최고
                day_env.append(norm.t_norm(rs[k]['TEMP_MIN_DAY'])) #최저
                day_env.append(rs[k]['SIE_HUMIDITY'] / 100) #습도
                day_env.append(norm.co2_norm(rs[k]['SIE_CO2'])) #CO2
                day_env.append(norm.hd_norm(rs[k]['SIE_HD'])) #HD



                date = datetime.datetime.strptime(str(rs[k]['FD_DAY_KEY']), datetime_format)
                dsin, dcos, _, _ = norm.datetime_to_sincos(date)
                day_env.extend([dsin, dcos, date.date()])

                day_env.append(rs[k]['FD_FARM_IDX']) #farm idx

                seq.append(day_env)

            seqs.append(seq)

        return seqs

#    def train(self, dataset: Dataset):
#        pass

