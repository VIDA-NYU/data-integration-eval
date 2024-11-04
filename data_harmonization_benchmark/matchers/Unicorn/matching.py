import logging
from typing import Any, Dict, Optional

import pandas as pd
from unicorn_zero import TrainApp

logger = logging.getLogger(__name__)


def matching(
    usecase: str,
    usecase_path: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    top_k: int = 10,
    use_gpu: bool = False,
    config: Optional[Dict[str, Any]] = dict(
        pretrain=False,
        load=True,  # Set this to True for pre-trained model, False for zero-shot !!!
        model="deberta_base",
        modelname="Temp",
        ckpt_path="matchers/Unicorn/checkpoint",
        ckpt="UnicornPlus",
        valentine_output=True,
    ),
):
    data_dir = usecase_path
    data_name = usecase
    src_orig_file = source
    tgt_orig_file = target
    golden_mappings = f"{data_dir}/groundtruth.csv"

    unicorn = TrainApp(usecase_path=usecase_path, use_gpu=use_gpu, **config)

    matches = unicorn.main()

    # logger.critical(f"[MATCHES] {matches}")

    return matches
