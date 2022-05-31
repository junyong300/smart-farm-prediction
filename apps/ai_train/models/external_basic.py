import datetime
import databases
from pandas.core.frame import DataFrame
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.python.data.ops.dataset_ops import Dataset
from dbconn import DbConn
from model_option import ModelOption
import normalize as norm
from .base_model import BaseModel

class ExternalBasicModel(BaseModel):
    def __init__(self, option: ModelOption):
        super().__init__(option)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            input_width = 60 * 24 * 2 // 10 
            input_shape = (input_width, 6)

            input_layer = tf.keras.layers.Input(input_shape)
            lstm_layer = tf.keras.layers.LSTM(32, dropout=0.1, input_shape=input_shape)
            dense_layer = tf.keras.layers.Dense(144 * 2 * 2)
            out_layer = tf.keras.layers.Reshape((144 * 2, 2))

            self.model = tf.keras.Model(inputs=input_layer, outputs=out_layer(dense_layer(lstm_layer(input_layer))))

            self.model.compile(
                loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=0.0005),
                metrics=[tf.metrics.MeanAbsoluteError()]
            )

            self.model.summary()
            # tfjs.converters.save_keras_model(self.model, "models_dir")

    async def fetchDb(self):
        conn = await self.getConn()

        target_devices = ','.join(map(str, self.option.devices))
        #sql = ("select device_idx didx, sensing_dt sdt, sie_temp t, sie_humidity h, sie_co2 co2 from sdh_internal "
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
    
    def preprocess(self, rs):
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
                # label_seq = [x[0] for x in seq[i + input_width:i + input_width + label_width: 10]]
                label_seq = [[x[0], x[1]] for x in seq[i + input_width:i + input_width + label_width: 10]]
                if len(label_seq) == label_width // 10:
                    refined_seqs.append((input_seq, label_seq))
        # refined_seqs = sorted(refined_seqs, key=lambda seq: seq[0][0][3])

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
            rt = [t, h] + tsn
            seq.append(rt)

        return seq
