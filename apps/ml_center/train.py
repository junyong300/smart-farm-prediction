from ml_common import setup
import models
import asyncio

setup("ml_center.log")

async def main():
    model = models.create("InternalSelf")
    ds = await model.makeDataset()
    model.train(ds)

asyncio.run(main())