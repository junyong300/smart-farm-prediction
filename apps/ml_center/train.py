import models
from ml_common.logger import setup

setup("ml.log")

model = models.create("InternalSelf")
ds = model.makeDataset()
model.train(ds)