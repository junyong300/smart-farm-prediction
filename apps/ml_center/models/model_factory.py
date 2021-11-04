from models.base_model import BaseModel
from models.internal_self import InternalSelfModel

def create(modelName: str) -> InternalSelfModel:
    className = modelName + "Model"
    model = globals()[className]
    return model()
