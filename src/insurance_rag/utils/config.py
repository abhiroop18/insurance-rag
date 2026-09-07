from pathlib import Path

import yaml

from insurance_rag.utils.config_types import AppConfig


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_config() -> AppConfig:

    with open(
        PARAMS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        raw_config = yaml.safe_load(f)

    return AppConfig.model_validate(
        raw_config
    )