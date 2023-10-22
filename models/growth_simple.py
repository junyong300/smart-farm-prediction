import pymysql
import pandas as pd
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
import datetime
import numpy as np

from dbconn import DbConn
from .base_model import BaseModel
from model_option import ModelOption
import normalize as norm

normBreedDict = {26: 0, 28: 1} # 26: 설향, 28: 흑토마토
gps = ['plant_height', 'stem_diameter', 'leaf_len', 'leaf_width', 'harvest_weight',
       'plant_height_nw', 'stem_diameter_nw', 'leaf_len_nw', 'leaf_width_nw', 'harvest_weight_nw']

class GrowthSimpleModel(BaseModel):
    '''
    생육수확이 급하여 최대한 단순하게 만든 모델

    * input1 growth: 생육, 수확
        생육수확 - 초장, 줄기굵기, 잎길이, 잎폭, 수확과중
        현 작기 week 차수
        작물코드 one hot

    * input2 env:
        4주간 1시간 평균 온도, 습도. sdh_internal에서 aggregation해서 가져온다
        [year sin, year cos, t, h] * 24 * 28일

    * output: d+7 생육. 초장, 줄기굵기, 잎길이, 잎폭
    '''

    def __init__(self, option: ModelOption):
        super().__init__(option)

        with self.strategy.scope():
            env_input_l = tf.keras.layers.Input((24 * 14, 6))
            env_input_reshape_l = tf.keras.layers.Reshape((24*14, 6, 1))(env_input_l)
            growth_input_l = tf.keras.layers.Input(10)
            #reshape_l = tf.keras.layers.Reshape((2,))

            conv_l = tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=(24*14, 6, 1))(env_input_reshape_l)
            faltten_l = tf.keras.layers.Flatten()(conv_l)
            env_dense_l = tf.keras.layers.Dense(32)(faltten_l)
            growth_dense_l = tf.keras.layers.Dense(8)(growth_input_l)
            concat_l = tf.keras.layers.concatenate([env_dense_l, growth_dense_l])
            out_dense = tf.keras.layers.Dense(5)(concat_l)
            #out_l = reshape_l(out_dense)
            out_l = out_dense

            self.model = tf.keras.Model(inputs=[env_input_l, growth_input_l], outputs=out_l)

            self.model.compile(
                # TODO: loss function을 현재 시각과 가까운 데이터에 가중치를 주도록 custom 할 필요가 있다
                loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.metrics.MeanAbsoluteError()]
            )


        self.model.summary()

    def makeDataset(self) -> Dataset:
        dfGrowth = self.loadOrFetch('fetchGrowthFromGjDb', self.fetchGrowthFromGjDb, "59.3.234.79", 3307, "cntd_farm_db", "root", ".fc12#$")
        dfGrowth = dfGrowth.drop(['idx', 'create_user', 'update_user', 'create_dt', 'update_dt'], axis=1)

        farmIdxes = dfGrowth['farm_idx'].drop_duplicates().tolist()
        farms = ','.join(map(str, farmIdxes))
        # 전체 생육조사기간 구하기
        minDt = dfGrowth['inv_dt'].min()
        minDt = minDt - datetime.timedelta(14)
        maxDt = dfGrowth['inv_dt'].max()

        dfEnv = self.loadOrFetch('fetchEnvFromGjDb', self.fetchEnvFromGjDb, "59.3.234.79", 3307, "cntd_farm_db", "root", ".fc12#$", farms, minDt, maxDt)

        input_env, input_growth, out = self.preprocessDbRs(dfGrowth, dfEnv)
        #input_env_tensor = tf.constant(np.array(input_env).astype(np.float32))
        input_env_tensor = tf.constant(input_env)
        input_growth_tensor = tf.constant(input_growth)
        out_tensor = tf.constant(out)

        dataset = tf.data.Dataset.from_tensor_slices(
            ((input_env_tensor, input_growth_tensor), out_tensor))

        return dataset

    def fetchGrowthFromGjDb(self, host, port, db, user, password):
        ''' 광주버전 db에서 생육데이터를 가져온다 '''
        conn = DbConn().connect('mysql', host, port, db, user, password)
        cur = conn.cursor(pymysql.cursors.DictCursor)

        # 생육 데이터
        '''
        sql = (
            "select fcs.farm_idx farmIdx, cb.crop_cd cropCd, "
            "glog.smpl_no sampleId, glog.invt_dt invDt, glog.week_idx week, "
            "glog.init_lnth plantHeight, glog.stem_thck stemDiameter, "
            "glog.leaf_lnth leafLength, glog.leaf_wdth leafWidth, glog.leaf_cnts leafCount, "
            "flog.blmg_cnts flowerCount, flog.frut_cnts fruitCount, flog.hrvt_weit harvestWeight "
            "from growth_log glog, flower_log flog, farm_cropping_season fcs, crop_breed_list cb "
            "where glog.fcs_idx = fcs.fcs_idx "
            "and flog.glog_idx = glog.glog_idx "
            "and fcs.crop_breed_idx = cb.crop_breed_idx "
            "order by glog.fcs_idx, smpl_no, invt_dt"
        )
        sql = (
            "select fcs.farm_idx, cb.crop_cd, cb.BREED_CD, glog.* from growth_log glog "
            "inner join farm_cropping_season fcs using (fcs_idx) "
            "inner join crop_breed_list cb using (crop_breed_idx) "
            "left join  "
            "    (select glog_idx, sum(blmg_cnts) blmg_cnts, sum(frut_cnts) frut_cnts, sum(hrvt_cnts) hrvt_cnts, sum(hrvt_weit) hrvt_weit  "
            "        from flower_log group by GLOG_IDX) flog "
            "on glog.GLOG_IDX = flog.glog_idx "
            "order by smpl_no "
        )
        '''
        sql = (
            "select * from growth "
            "order by fcs_idx, sample_id, inv_dt "
        )

        cur.execute(sql)
        rsGrowth = cur.fetchall()
        return rsGrowth

    def fetchEnvFromGjDb(self, host, port, db, user, password, farms, minDt, maxDt):
        conn = DbConn().connect('mysql', host, port, db, user, password)
        cur = conn.cursor(pymysql.cursors.DictCursor)

        # 환경 데이터
        sql = (
            "select fd.farm_idx farmIdx, cast(date_format(sensing_dt , '%Y-%m-%d %H:00:00') as datetime) sdt, avg(sie_temp) t, avg(sie_humidity) h from sdh_internal s, farm_dong fd "
            F"where farm_idx in ({farms}) "
            F"and sensing_dt between '{minDt}' and '{maxDt}' "
            "and s.device_idx = fd.in_env_device_idx "
            "group by farm_idx, sdt "
        )

        cur.execute(sql)
        rsEnv = cur.fetchall()
        return rsEnv

    def preprocessDbRs(self, dfGrowth: DataFrame, dfEnv: DataFrame):
        ''' DB에서 가져 온 recordset의 처리 '''

        dfEnv['sdtDate'] = dfEnv['sdt'].dt.date
        grGrowthFarms = dfGrowth.groupby('farm_idx')

        input_env = []
        input_growth = []
        out = []

        for farmIdx, grGrowthFarm in grGrowthFarms:
            # grGrowthFarm: 개별 농장의 모든 growthes

            # farmIdx 농장의 모든 일자 환경 데이터
            def addEnv(grp: DataFrame):
                dfFarmEnv = dfEnv.loc[dfEnv['farmIdx'] == farmIdx]
                invDt = grp.name
                startDate = invDt + datetime.timedelta(days=-14)
                endDate = invDt + datetime.timedelta(days=-1)
                dfEnvRange = dfFarmEnv[(dfFarmEnv['sdtDate'] >= startDate) & (dfFarmEnv['sdtDate'] <= endDate)]
                envValues = dfEnvRange.drop(['farmIdx', 'sdtDate'], axis=1).values.tolist()
                grp['env'] = [envValues] * len(grp)
                return grp

            # invDt별 group 으로 나눠서 각 그룹에 대해 env를 구한다
            grGrowthFarmInvDts = grGrowthFarm.groupby('inv_dt', group_keys=False)
            grGrowthFarmInvDts = grGrowthFarmInvDts.apply(addEnv)

            # 작기의 sample별로 그룹으로 나누고, 다음 조사를 이전 조사와 합친 후, 다음 조사가 없으면 버린다
            def addNext(grp: DataFrame):
                nextWeek = grp.drop(['fcs_idx', 'sample_id', 'farm_idx', 'crop_idx', 'breed_idx'], axis=1).shift(-1)
                grp = grp.join(nextWeek, rsuffix='_nw')
                grp = grp[grp['inv_dt_nw'].notnull()]
                grp['days'] = (grp['inv_dt_nw'] - grp['inv_dt']) / np.timedelta64(1, 'D')
                grp = grp[grp['days'] < 22] # 3주 이상 차이가 나면 버린다
                return grp

            grGrowthFarmSamples = grGrowthFarmInvDts.groupby(['farm_idx', 'fcs_idx', 'sample_id'], group_keys=False)
            grGrowthFarmSamples = grGrowthFarmSamples.apply(addNext)

            # normalize
            df = grGrowthFarmSamples
            df[gps] = df[gps].fillna(0)

            for index, r in df.iterrows():
                daySin, dayCos, yearSin, yearCos = norm.datetime_to_sincos(datetime.datetime.combine(r['inv_dt'], datetime.datetime.min.time())) # date to datetime, and normalize
                nDays = ((r['days']) - 7) / 21 # 위에서 최대 32주 텀의 조사까지만 필터링 했으므로
                nBreed = normBreedDict[r['breed_idx']]
                nWeek = r['week'] / 50 # 최대 50주로 간주

                nPlantHeight = (r['plant_height'] - 132) / 74
                nStemDiameter = (r['stem_diameter'] - 7.7) / 1.19
                nLeafLen = (r['leaf_len'] - 19) / 13
                nLeafWidth = (r['leaf_width'] - 15) / 11
                nHarvestWeight = (r['harvest_weight'] - 62) / 72

                nPlantHeightNw = (r['plant_height_nw'] - 132) / 74
                nStemDiameterNw = (r['stem_diameter_nw'] - 7.7) / 1.19
                nLeafLenNw = (r['leaf_len_nw'] - 19) / 13
                nLeafWidthNw = (r['leaf_width_nw'] - 15) / 11
                nHarvestWeightNw = (r['harvest_weight_nw'] - 62) / 72

                nEnv = []
                if (len(r['env']) < 24 * 14):
                    nEnv = [[0, 0, 0, 0, 0, 0]] * (24 * 14 - len(r['env']))
                for env in r['env']:
                    envDSin, envDCos, envYSin, envYCos = norm.datetime_to_sincos(env[0])
                    nEnv.append([envDSin, envDCos, envYSin, envYCos, norm.t_norm(env[1]), norm.h_norm(env[2])])

                input_env.append(nEnv)
                input_growth.append([nBreed, yearSin, yearCos, nWeek, nDays, nPlantHeight, nStemDiameter, nLeafLen, nLeafWidth, nHarvestWeight])
                out.append([nPlantHeightNw, nStemDiameterNw, nLeafLenNw, nLeafWidthNw, nHarvestWeightNw])


#            df_input_env = df[['inv_dt', 'env']]
#            df_input_growth = df[['breed_idx', 'days', 'plant_height', 'harvest_weight']]
#            df_output = df[['plant_height_n', 'harvest_weight_n']]
#            print(df_input_env.head())
#            print(df_input_growth.head())
#            print(df_output.head())

        return input_env, input_growth, out


#    def train(self, dataset: Dataset):
#        pass
