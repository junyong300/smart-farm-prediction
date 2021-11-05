import tensorflow as tf
import pandas as pd
import datetime

from pandas.core.frame import DataFrame
from ml_common.redis_message import ModelOption
from models.base_model import BaseModel
import ml_common.normalize as norm

class InternalSelfModel(BaseModel):
    def __init__(self):
        super().__init__()
    
    def fetchDb(self, modelOption: ModelOption):
        deviceId = modelOption.deviceId
        if not deviceId:
            deviceId = 2090099

        try:
            etime = modelOption.time
            #etime = datetime.datetime.strptime(etime, '')
        except Exception:
            etime = datetime.datetime.now()
        
        stime = etime - datetime.timedelta(days=2)

        sql = ("select sensingtime, temp, co2 "
            "from env.internal "
            F"where deviceid = {deviceId} and sensingtime between '{stime}' and '{etime}' "
            "and temp between -50 and 80 "
            "and co2 between 0 and 5000 "
        )
        self.logger.debug(sql)
        self.pgConn.execute(sql)
        rs = self.pgConn.fetchall()
        return rs

    def makeInput(self, rs):
        base = rs[len(rs) - 1]['sensingtime'].minute

        df = pd.DataFrame(rs)
        df = df.set_index('sensingtime')
        df: DataFrame = df.resample('10T', base=base).bfill()
        df = df.drop(index=df.index[:df.shape[0] - 288]) # drop rows over 288

        df_input = pd.DataFrame()
        df_input[['temp', 'co2']] = df.apply(lambda x: (norm.t_norm(x['temp']), norm.co2_norm(x['co2'])), axis=1, result_type="expand")
        df_input[['dsin', 'dcos', 'ysin', 'ycos']] = df.index.to_frame().apply(lambda x: norm.datetime_to_sincos(x['sensingtime']), axis=1, result_type="expand")


        input_tensor = tf.constant(df_input)
        input_tensor = tf.reshape(input_tensor, [-1, 288, 6])
        return input_tensor, df

    def predict(self, model: tf.keras.Model, modelOption: ModelOption):
        rs = self.fetchDb(modelOption)
        input, df = self.makeInput(rs)
        [pred] = model.predict(input)
        pred = [norm.t_denorm(x) for x in pred]

        df.insert(0, 'sensingTime', df.index)
        df['sensingTime'] = df['sensingTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        input_list = df.values.tolist()

        self.logger.debug("predict done")
        return pred, input_list
