from abc import *
from tensorflow.keras import Model
from ml_common import BaseModel, ModelOption

class InferModel(BaseModel, metaclass=ABCMeta):
    def __init__(self, kerasModel):
        self.model = kerasModel

    @abstractmethod
    async def predictFromDb(self, dbConn, modelOption: ModelOption):
        pass
