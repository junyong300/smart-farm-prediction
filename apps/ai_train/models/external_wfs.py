import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
import datetime
import os
from plot import saveFig
from model_option import ModelOption
import normalize as norm
from .base_model import BaseModel

class ExternalWfsModel(BaseModel):
    def __init__(self, option: ModelOption):
        super().__init__(option)

        with self.strategy.scope():
            ext_input_width = 60 * 24 * 2 // 10 
            ext_input_shape = (ext_input_width, 6)
            wfs_input_width = 16
            wfs_input_shape = (wfs_input_width, 6)
            label_width = 60 * 24 * 2 // 10
            label_shape = (label_width, 2)

            ext_input_layer = tf.keras.layers.Input(ext_input_shape)
            wfs_input_layer = tf.keras.layers.Input(wfs_input_shape)

            lstm_layer = tf.keras.layers.LSTM(288, dropout=0.1, input_shape=ext_input_shape)
            ext_dense_layer = tf.keras.layers.Dense(288 * 2)

            wfs_dense_layer = tf.keras.layers.Dense(64)
            wfs_out_layer = wfs_dense_layer(wfs_input_layer)
            wfs_out_layer = tf.keras.layers.Flatten()(wfs_out_layer)

            concat_layer = tf.keras.layers.concatenate([ext_dense_layer(lstm_layer(ext_input_layer)), wfs_out_layer])
            dense_layer1 = tf.keras.layers.Dense(512)
            dense_layer2 = tf.keras.layers.Dense(label_width * 2)
            reshape_layer = tf.keras.layers.Reshape(label_shape)
            output_layer = reshape_layer(dense_layer2(dense_layer1(concat_layer)))

            self.model = tf.keras.Model(inputs=[ext_input_layer, wfs_input_layer], outputs=output_layer)

            self.model.compile(
                loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.metrics.MeanAbsoluteError()]
            )

        self.model.summary()
    
    async def makeDataset(self) -> Dataset:
        rs = await self.loadOrFetch()
        ext_seqs, wfs_seqs = self.preprocess(rs)
        self.logger.info("external-wfs model preprocessing done")

        # make dataset
        input_seqs, label_seqs = zip(*ext_seqs)
        input_seqs = np.array(input_seqs)
        input_seqs = np.delete(input_seqs, [0], axis=2) # sdt 삭제
        ext_input_tensor = tf.constant(input_seqs)
        label_tensor = tf.constant(label_seqs)

        for s in wfs_seqs:
            if (len(s) != 16):
                print(len(s))
        wfs_input_tensor = tf.constant(wfs_seqs)

        dataset = tf.data.Dataset.from_tensor_slices(
            ((ext_input_tensor, wfs_input_tensor), label_tensor))
        return dataset

    async def loadOrFetch(self):
        """ external과 wfs 각각에 대해 처리한다 """

        # external
        # feather 파일이 있으면 사용하고, 없으면 db에서 불러 와서 feather 파일로 저장
        fileName = os.path.join("temp", self.id + ".feather")
        if os.path.isfile(fileName):
            df_ext = pd.read_feather(fileName)
        else:
            rs = await self.fetchDb()
            df_ext = pd.DataFrame(rs)
            df_ext.columns = rs[0].keys()
            if not os.path.exists("temp"):
                os.makedirs("temp")
            
            df_ext.to_feather(fileName)

        # wfs
        # feather 파일이 있으면 사용하고, 없으면 db에서 불러 와서 feather 파일로 저장
        fileName = os.path.join("temp", self.id + "_wfs.feather")
        if os.path.isfile(fileName):
            df_wfs = pd.read_feather(fileName)
        else:
            rs = await self.fetchWfs()
            df_wfs = pd.DataFrame(rs)
            df_wfs.columns = rs[0].keys()
            if not os.path.exists("temp"):
                os.makedirs("temp")
            
            df_wfs.to_feather(fileName)

        # wfs에서 청양 지역만 필터링
        df_wfs = df_wfs.loc[(df_wfs.wth_st_grid_x == 59)
                    & (df_wfs.wth_st_grid_y == 99), :]
        df_wfs = df_wfs.drop(
            ['wth_st_grid_x', 'wth_st_grid_y'], axis=1)

        wfs_start_dt = df_wfs.iloc[0, 0]
        wfs_end_dt = df_wfs.iloc[len(df_wfs) - 1, 0]

        df_ext = df_ext.loc[((df_ext.sdt + datetime.timedelta(days=3)) >= wfs_start_dt)
                            & (df_ext.sdt - datetime.timedelta(days=2) <= wfs_end_dt), :]

        self.logger.info("Model recordset loaded")

        return (df_ext, df_wfs)


    async def fetchDb(self):
        conn = await self.getConn()

        target_devices = ','.join(map(str, self.option.devices))
        sdate = self.option.sdate.replace("-", "")
        edate = self.option.edate.replace("-", "")

        sql = ("select device_idx didx, sensing_dt sdt, sews_temp t, sews_humidity h from sdh_external "
                F"where device_idx in ({target_devices}) "
                # F"and daykey between {sdate} and {edate} "
                "and sews_temp between -50 and 50 "
                "and sews_humidity between 0 and 100 "
                "order by didx, sdt"
        )

        self.logger.debug("sql:" + sql)
        rs = await conn.fetch_all(sql)
        await conn.disconnect()

        return rs
    
    async def fetchWfs(self):
        conn = await self.getConn()
        sql = (
            "SELECT TIMESTAMP(base_date, base_time) bdt, TIMESTAMP(fcst_date, fcst_time) fdt, wth_st_grid_x, wth_st_grid_y, t3h, pop "
            "FROM weather_forecast_space "
            "HAVING TIMESTAMPDIFF(HOUR, bdt, fdt) <= 51 "   # 48 + 3시간
            "ORDER BY wth_st_grid_x, wth_st_grid_y, bdt, fdt "
            # "LIMIT 10000 "
        )

        rs = await conn.fetch_all(sql)
        await conn.disconnect()
        return rs

    def preprocess(self, rs):
        ext_seqs_all = self.preprocess_ext(rs[0])
        df_wfs = rs[1]

        ext_seqs = []
        ext_input_width = 60 * 24 * 2
        gr_wfs = df_wfs.groupby(['bdt'])
        wfs_seqs = []
        wfs_idx = 0
        grs = [group for bdt, group in gr_wfs]
        bts = 0
        lastSts = 0

        for i, seq in enumerate(ext_seqs_all):
            sts = seq[0][ext_input_width // 10 - 1][0]
            if lastSts > sts: # ext_seqs_all 에서 device가 바뀌면 다시 과거 날짜로 돌아가므로 wfs도 처음으로 되돌린다
                wfs_idx = 0
            bts = datetime.datetime.timestamp(grs[wfs_idx]['bdt'].iloc[0])

            # bdt가 sts보다 3시간 이전이면 아니게 될 때까지 wfs 건너 뜀
            while len(grs[wfs_idx]) != 16 or (bts + 3600 * 3 < sts + 600) :
                wfs_idx += 1
                if wfs_idx >= len(grs):
                    break
                bts = datetime.datetime.timestamp(grs[wfs_idx]['bdt'].iloc[0])
            
            # print(datetime.datetime.fromtimestamp(sts), datetime.datetime.fromtimestamp(bts))

            # 필요한 예보인 경우
            if bts <= sts and (bts + 3600 * 3) > sts:
                wts_seq = [[norm.t_norm(t3h), pop / 100 / 2] + [item for item in norm.datetime_to_sincos(fdt)] # pop의 비중을 절반으로 줄임
                           for t3h, pop, fdt in grs[wfs_idx][['t3h', 'pop', 'fdt']].values]
                wfs_seqs.append(wts_seq)
                ext_seqs.append(ext_seqs_all[i])
            
            lastSts = sts

        return ext_seqs, wfs_seqs

    def preprocess_ext(self, rs):
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
            h = hr / 100 / 2
            #l = norm.l_norm(lr)
            #co2 = norm.co2_norm(co2r)
            tsn = list(norm.timestamp_to_sincos(ts))

            #rt = [tr, hr, co2r, ts, t, h, co2] + tsn
            rt = [ts, t, h] + tsn
            seq.append(rt)

        return seq

    def test(self, dataset: Dataset):
        self.logger.info("Test start")

        testset = dataset.skip(len(dataset) - 1000)
        self.logger.info(F"Test Size: {len(testset)}")

        testset = testset.batch(512)
        pred = self.predict(testset)
        pred_t = pred[:,:,0]
        pred_t = norm.t_denorm(pred_t)

        inputs_ds = testset.map(lambda i, l: i[0]).unbatch()
        inputs = np.stack(list(inputs_ds))
        inputs = inputs[:,:,0]
        inputs = norm.t_denorm(inputs)

        labels_ds = testset.map(lambda i, l: l).unbatch()
        labels = np.stack(list(labels_ds))
        labels = labels[:,:,0]
        labels = norm.t_denorm(labels)

        saveFig(inputs, pred_t, labels, self.option.modelId, 20)
