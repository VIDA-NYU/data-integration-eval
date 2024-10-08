import logging
import os
import time
from typing import Union

import pandas as pd

GDC_DATA_PATH = os.path.join(os.path.dirname(__file__), "./resource/gdc_table.csv")

logger = logging.getLogger(__name__)


class Config:
    def __init__(
        self,
        usecase: str,
        source: Union[str, pd.DataFrame],
        target: Union[str, pd.DataFrame],
        ground_truth: Union[str, pd.DataFrame],  # "source" | "target"
        scorer: str = "accuracy",
        n_jobs: int = 1,
        top_k: int = 10,
    ):
        if isinstance(source, str):
            self.source = pd.read_csv(source)
        else:
            self.source = source

        if target in ["gdc"]:
            self.target = pd.read_csv(GDC_DATA_PATH)
        elif isinstance(target, str):
            self.target = pd.read_csv(target)
        else:
            self.target = target

        if isinstance(ground_truth, str):
            self.ground_truth = pd.read_csv(ground_truth)
        else:
            self.ground_truth = ground_truth

        self.n_jobs = n_jobs
        self.usecase_name = usecase.split("/")[-1]
        self.usecase_path = usecase
        self.top_k = top_k

    def get_source(self) -> pd.DataFrame:
        return self.source

    def get_target(self) -> pd.DataFrame:
        return self.target

    def get_ground_truth(self) -> pd.DataFrame:
        return self.ground_truth

    def get_scorer(self) -> callable:
        def accuracy(ground_truth: list[str], matches: list[str]) -> float:
            return sum([1 for i, j in zip(ground_truth, matches) if i == j]) / len(
                ground_truth
            )

        return accuracy

    def get_ground_truth_set(self) -> set:
        gt_set = set()
        for row in self.ground_truth.itertuples():
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





