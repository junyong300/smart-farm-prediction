import logging

from model_option import ModelOption
from models.base_model import BaseModel
from .external_basic import ExternalBasicModel
from .external_wfs import ExternalWfsModel
from .internal_self import InternalSelfModel
from .internal_basic import InternalBasicModel
from .growth_basic import GrowthBasicModel
from .internal_poc import InternalPocModel
from .pest_basic import PestBasicModel

logger = logging.getLogger(__name__)

# def create(option: ModelOption) -> InternalSelfModel or ExternalBasicModel:
def create(option: ModelOption) -> BaseModel:
    className = option.model + "Model"
    try:
        model = globals()[className]
        return model(option)
    except Exception:
        logger.exception("Failed to load model: " + className)
        return None
