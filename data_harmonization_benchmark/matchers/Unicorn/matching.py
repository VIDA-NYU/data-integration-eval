from unicorn_zero import TrainApp
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def matching(
    usecase: str,
    usecase_path: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    top_k: int = 10,
    config: Optional[Dict[str, Any]] = dict(
        pretrain=False,
        load=True,
        model="deberta_base",
        modelname="UnicornZero",
    ),
):
    data_dir=usecase_path
    data_name=usecase
    src_orig_file=source
    tgt_orig_file=target
    golden_mappings=f"{data_dir}/groundtruth.csv"
    
    unicorn = TrainApp(usecase_path=usecase_path, **config)

    unicorn.main()
    matches = unicorn.get_matches(top_k=top_k)

    logger.critical(f"[MATCHES] {matches}")

    return matches