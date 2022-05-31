import databases
from pandas.core.frame import DataFrame
from tensorflow.python.data.ops.dataset_ops import Dataset

from .base_model import BaseModel
from model_option import ModelOption

class GrowthBasicModel(BaseModel):
    def __init__(self, option: ModelOption):
        super().__init__(option)
    
    async def makeDataset(self) -> Dataset:
        rs = await self.loadOrFetch()
        seqs = self.preprocess(rs)
        self.logger.info("Preprocessing done")

        return None

    async def fetchDb(self, dbConn: databases):
        target_devices = ','.join(map(str, self.option.devices))
        #sql = ("select device_idx didx, sensing_dt sdt, sie_temp t, sie_humidity h, sie_co2 co2 from sdh_internal "
        sdate = self.option.sdate.replace("-", "")
        edate = self.option.edate.replace("-", "")
        sql = ("select device_idx didx, sensing_dt sdt, sews_temp t from sdh_external "
                F"where device_idx in ({target_devices}) "
                F"and daykey between {sdate} and {edate} "
                "and sews_temp between -50 and 50 "
                "order by didx, sdt"
        )
        self.logger.debug("sql:" + sql)
        rs = await dbConn.fetch_all(sql)
        return rs
    
    def preprocess(self, rs):
        input_width = 60 * 24 * 2
        label_width = 60 * 24
        min_point = input_width + label_width
        device_data_count = 0
        seq = []
        seqs = []

        if isinstance(rs, DataFrame):
            rs = rs.to_dict('records')

        for i in range(len(rs) - 1):
            pass

    def train(self, dataset: Dataset):
        pass

    