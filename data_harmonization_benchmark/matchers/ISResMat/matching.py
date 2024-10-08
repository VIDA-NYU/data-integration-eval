from ISResMat import TrainApp
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
        n_trn_cols=200,
        batch_size=1,
        frag_height=6,
        frag_width=12,
        learning_rate=3e-5,
    ),
):
    data_dir=usecase_path
    data_name=usecase
    src_orig_file=source
    tgt_orig_file=target
    golden_mappings=f"{data_dir}/groundtruth.csv"
    
    isresmat = TrainApp(
        col_name_prob=0,
        store_matches=0,
        comment="inst_001",
        dataset_name=data_name,
        orig_file_src=src_orig_file,
        orig_file_tgt=tgt_orig_file,
        orig_file_golden_matches=golden_mappings,
        process_mode=1,
        n_val_cols=1, **config)

    isresmat.do_trn(1)
    isresmat.eval_to_match()
    matches = isresmat.get_matches(top_k=top_k)

    logger.critical(f"[MATCHES] {matches}")

    return matches