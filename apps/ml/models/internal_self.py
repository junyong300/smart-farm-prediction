import tensorflow as tf
import pandas as pd
import datetime
from pandas.core.frame import DataFrame

from ml_common.redis_message import ModelOption
import ml_common.normalize as norm
from models.infer_model import InferModel

class InternalSelfModel(InferModel):

    async def fetchDb(self, dbConn, modelOption: ModelOption):
        deviceId = modelOption.deviceId
        if not deviceId:
            deviceId = 2090098

        try:
            etime = modelOption.time
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
        rs = await dbConn.fetch_all(sql)
        return rs

    def makeInput(self, rs):
        if len(rs) == 0:
            return None, None

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

    async def predictFromDb(self, dbConn, modelOption: ModelOption):
        rs = await self.fetchDb(dbConn, modelOption)
        input, df = self.makeInput(rs)
        if input is None:
            return [], []

        [pred] = self.model.predict(input)
        pred = [norm.t_denorm(x) for x in pred]

        df.insert(0, 'sensingTime', df.index)
        df['sensingTime'] = df['sensingTime'].dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m-%d %H:%M:%S')
        df = df[['sensingTime', 'temp', 'co2']]
        input_list = df.values.tolist()

        self.logger.debug("predict done")
        return pred, input_list
    