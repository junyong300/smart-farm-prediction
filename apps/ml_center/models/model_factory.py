from models.internal_self import InternalSelfModel

def create(modelName: str) -> InternalSelfModel:
    className = modelName + "Model"
    model = globals()[className]
    return model()
