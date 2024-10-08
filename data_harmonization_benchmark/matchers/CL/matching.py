from typing import Any, Dict, Optional

import pandas as pd
from bdikit.mapping_algorithms.column_mapping.topk_matchers import CLTopkColumnMatcher
from bdikit.models.contrastive_learning.cl_api import DEFAULT_CL_MODEL


def matching(
    usecase: str,
    usecase_path: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    top_k: int = 10,
    config: Optional[Dict[str, Any]] = dict(),
):
    matcher = CLTopkColumnMatcher(DEFAULT_CL_MODEL)

    matches = matcher.get_recommendations(source, target, top_k=top_k, **config)

    output = {}
    for match in matches:
        source = match["source_column"]
        for candidate in match["top_k_columns"]:
            candidate_name = candidate.column_name
            candidate_score = candidate.score
            output[(("source", source), ("target", candidate_name))] = candidate_score
    return output
