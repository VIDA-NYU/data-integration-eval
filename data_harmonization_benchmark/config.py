import logging
import os
import time
from typing import Union, Optional

import pandas as pd

GDC_DATA_PATH = os.path.join(os.path.dirname(__file__), "./resource/gdc_table.csv")

logger = logging.getLogger(__name__)


class Config:
    def __init__(
        self,
        usecase: str,
        subtasks: list[str],
        sources: list[Union[str, pd.DataFrame]],
        targets: list[Union[str, pd.DataFrame]],
        ground_truths: list[Union[str, pd.DataFrame]],  # "source" | "target"
        scorer: str = "accuracy",
        n_jobs: int = 1,
        top_k: int = 10,
    ):
        self.subtasks = subtasks
        self.sources = iter(sources)

        if targets[0] in ["gdc"]:
            self.targets = pd.read_csv(GDC_DATA_PATH)
        else:
            self.targets = iter(targets)

        self.ground_truths = iter(ground_truths)

        self.n_jobs = n_jobs
        self.usecase_name = usecase.split("/")[-1]
        self.usecase_path = usecase
        self.top_k = top_k

    def get_source(self) -> Optional[pd.DataFrame]:
        source = next(self.sources)
        if isinstance(source, pd.DataFrame):
            return source
        if os.path.exists(source):
            return pd.read_csv(source)
        return None

    def get_target(self) -> Optional[pd.DataFrame]:
        if isinstance(self.targets, pd.DataFrame):
            return self.targets
        target = next(self.targets)
        if isinstance(target, pd.DataFrame):
            return target
        if os.path.exists(target):
            return pd.read_csv(target)
        return None

    def get_ground_truth(self) -> Optional[pd.DataFrame]:
        ground_truth = next(self.ground_truths)
        if isinstance(ground_truth, pd.DataFrame):
            return ground_truth
        if os.path.exists(ground_truth):
            return pd.read_csv(ground_truth)
        return None

    def get_scorer(self) -> callable:
        def accuracy(ground_truth: list[str], matches: list[str]) -> float:
            return sum([1 for i, j in zip(ground_truth, matches) if i == j]) / len(
                ground_truth
            )

        return accuracy

    def get_ground_truth_set(self) -> set:
        ground_truth = self.get_ground_truth()
        if ground_truth is None:
            logger.error("[Config] get_ground_truth_set ground_truth is invalid!")
            return set()
        gt_set = set()
        for row in ground_truth.itertuples():
            gt_set.add((row.source, row.target))

        return gt_set

    def get_n_jobs(self) -> int:
        return self.n_jobs

    def get_usecase_name(self) -> str:
        return self.usecase_name

    def get_usecase_path(self) -> str:
        return self.usecase_path

    def get_top_k(self) -> int:
        return self.top_k

    def get_subtasks(self) -> list[str]:
        return self.subtasks
