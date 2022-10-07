import json
from types import SimpleNamespace

from config import Config

class ModelOption:
    def __init__(self, modelId, conf: Config):
        self.conf = conf;

        fileName = F'ai_models/train_options/{modelId}.json'
        f = open(fileName)
        x = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

        self.modelId = modelId
        for key in x.__dict__:
            setattr(self, key, x.__dict__[key])

    def get(self, key, default=None):
        return getattr(self, key) if hasattr(self, key) else default

    def __str__(self):
        return F"{self.model}/{self.modelId}"