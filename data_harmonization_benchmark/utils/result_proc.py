import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def save_result_csv(df: pd.DataFrame, path: str, append: bool = False) -> None:
    mode = "w"
    if append:
        mode = "a"

    if os.path.exists(path):
        header = False
    else:
        header = True

    df.to_csv(path, mode=mode, header=header, index=False)
    logger.info("[utils]Results appended to %s", path)


def parse_results(source, target, matcher, runtime, all_metrics, output):
    results = {
        "source": [source],
        "target": [target],
        "matcher": [matcher],
        "runtime": [runtime],
    }

    for metrix_name, score in all_metrics.items():
        results[metrix_name] = score

    result_df = pd.DataFrame(results)

    save_result_csv(result_df, output, append=True)
