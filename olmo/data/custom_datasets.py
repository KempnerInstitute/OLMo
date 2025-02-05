import importlib
import logging
from typing import Optional, Tuple

from torch.utils.data import Dataset

from ..config import DataConfig
from ..exceptions import OLMoConfigurationError

__all__ = ["build_custom_dataset"]

LOGGER = logging.getLogger(__name__)

def build_custom_dataset(data_config: DataConfig) -> Dataset:
    assert data_config.custom_dataset
    if not data_config.custom_dataset.name:
        raise OLMoConfigurationError("custom_dataset_class is required when using a custom dataset")
    LOGGER.warning("Using custom dataset class, deterministic training is not guaranteed")
    LOGGER.info(f"Loading custom dataset {data_config.custom_dataset.name} from module {data_config.custom_dataset.module}")
    dataset_class = data_config.custom_dataset.name
    dataset_module = data_config.custom_dataset.module
    if not dataset_module:
        dataset_module, dataset_class = extract_module_and_class(dataset_class)
    if dataset_module is None:
        raise OLMoConfigurationError(
            "when using custom_dataset_class, use the full module path of the class or specify custom_dataset_module"
        )
    module = importlib.import_module(dataset_module)
    dataset_class = getattr(module, dataset_class)
    return dataset_class(**data_config.custom_dataset.args)  # type: ignore


def extract_module_and_class(name: str) -> Tuple[Optional[str], str]:
    class_module = name.split(".")
    if len(class_module) < 2:
        return None, class_module[0]
    return ".".join(class_module[:-1]), class_module[-1]

