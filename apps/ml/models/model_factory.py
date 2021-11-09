from typing import Union
from models.infer_model import InferModel
from models.internal_self import InternalSelfModel

def create(modelName: str, kerasModel) -> Union[InferModel, InternalSelfModel]:
    className = modelName + "Model"
    model = globals()[className]
    return model(kerasModel)
