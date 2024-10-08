import os
import sys
import logging
import time
import importlib.util
from typing import Tuple, Union, Any

from config import Config
from valentine.algorithms.matcher_results import MatcherResults

from utils.mrr import compute_mean_ranking_reciprocal

logger = logging.getLogger(__name__)
MATCHERS_PATH = "./matchers"

def get_matcher(matcher_name: str):
    matchers_scan = [f.name for f in os.scandir(MATCHERS_PATH) if f.is_dir() and not f.name.startswith(".")]
    if matcher_name not in matchers_scan:
        logger.error("Matcher not exist!")
        return None
    sys.path.append(os.path.join(MATCHERS_PATH, matcher_name))
    sys.path.append(os.path.join(MATCHERS_PATH, matcher_name))
    spec = importlib.util.spec_from_file_location(f"matchers.{matcher_name}", os.path.join(MATCHERS_PATH, f"{matcher_name}/matching.py"))
    matcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher)

    return matcher

def parse_matches_to_valentine(matches):
    if isinstance(matches, MatcherResults):
        return matches
    return MatcherResults(matches)

def matching(
    config: Config, matcher: str
) -> Tuple[Any, int]:
    ground_truth = config.get_ground_truth().sort_values(by="source")
    source = config.get_source()[ground_truth["source"].values]
    target = config.get_target()

    matcher = get_matcher(matcher)

    scores = []
    runtimes = []
    n_jobs = config.get_n_jobs()
    for _ in range(n_jobs):
        # record start time
        start = time.time()

        # matches: "source" | "target"
        matches = matcher.matching(config.get_usecase_name(),
                                   config.get_usecase_path(),
                                   config.get_source(),
                                   config.get_target(),
                                   config.get_top_k(),
                                  )

        # record end time
        end = time.time()

        # matches = matches.sort_values(by="source")

        # score = config.get_scorer()(
        #     list(ground_truth["target"].values), list(matches["target"].values)
        # )

        matches = parse_matches_to_valentine(matches)
        if matches is None:
            logger.error("Matches is not in MatcherResults")
            return

        mrr_score = compute_mean_ranking_reciprocal(matches, config.get_ground_truth_set())

        all_metrics = matches.get_metrics(config.get_ground_truth_set())
        
        one2one_metrics = matches.one_to_one().get_metrics(config.get_ground_truth_set())


        all_metrics["MRR"] = mrr_score
        for metrix_name, score in one2one_metrics.items():
            all_metrics[f"one2one_{metrix_name}"] = score
        
        # scores.append(score)
        runtimes.append((end - start) * 1e3)

    return all_metrics, int(sum(runtimes) / n_jobs)
