from typing import Any, Dict, Optional

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import DistributionBased


def matching(
    usecase: str,
    usecase_path: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    top_k: int = 10,
    use_gpu: bool = False,
    config: Optional[Dict[str, Any]] = dict(),
):
    matcher = DistributionBased(**config)

    matches = valentine_match(source, target, matcher)

    return matches
