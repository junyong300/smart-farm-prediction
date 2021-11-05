from typing import Union
from models.base_model import BaseModel
from models.internal_self import InternalSelfModel

def create(modelName: str) -> Union[BaseModel, InternalSelfModel]:
    className = modelName + "Model"
    model = globals()[className]
    return model()
