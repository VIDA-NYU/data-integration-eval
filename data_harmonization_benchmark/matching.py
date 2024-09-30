import logging
import time
from typing import Tuple, Union

from bdikit import BaseSchemaMatcher, match_schema

from config import Config

logger = logging.getLogger(__name__)


def matching(
    config: Config, matcher: Union[str, BaseSchemaMatcher]
) -> Tuple[float, int]:
    ground_truth = config.get_ground_truth().sort_values(by="source")
    source = config.get_source()[ground_truth["source"].values]
    target = config.get_target()

    scores = []
    runtimes = []
    n_jobs = config.get_n_jobs()
    for _ in range(n_jobs):
        # record start time
        start = time.time()

        # matches: "source" | "target"
        matches = match_schema(source, target, matcher)

        # record end time
        end = time.time()

        matches = matches.sort_values(by="source")

        score = config.get_scorer()(
            list(ground_truth["target"].values), list(matches["target"].values)
        )

        scores.append(score)
        runtimes.append((end - start) * 1e3)

    return sum(scores) / n_jobs, int(sum(runtimes) / n_jobs)
