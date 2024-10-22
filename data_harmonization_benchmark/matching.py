import importlib.util
import logging
import os
import sys
import time
import json
from typing import Any, Tuple, Union

from valentine.algorithms.matcher_results import MatcherResults

from config import Config
from utils.mrr import compute_mean_ranking_reciprocal

logger = logging.getLogger(__name__)
MATCHERS_PATH = "./matchers"
OUTPUT_PATH = "./tmp"


def get_matcher(matcher_name: str):
    matchers_scan = [
        f.name
        for f in os.scandir(MATCHERS_PATH)
        if f.is_dir() and not f.name.startswith(".")
    ]
    if matcher_name not in matchers_scan:
        logger.error("Matcher not exist!")
        return None
    sys.path.append(os.path.join(MATCHERS_PATH, matcher_name))
    sys.path.append(os.path.join(MATCHERS_PATH, matcher_name))
    spec = importlib.util.spec_from_file_location(
        f"matchers.{matcher_name}",
        os.path.join(MATCHERS_PATH, f"{matcher_name}/matching.py"),
    )
    matcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher)

    return matcher


def parse_matches_to_valentine(matches):
    if isinstance(matches, MatcherResults):
        return matches
    return MatcherResults(matches)

def save_json_results(results, file_path):
    save_file = os.path.join(OUTPUT_PATH, file_path)
    with open(save_file, 'w', encoding='utf-8') as f:
        result_parsed = {}
        for k, v in results.items():
            result_parsed[str(k)] = float(v)
        json.dump(result_parsed, f, ensure_ascii=False, indent=4)



def matching(config: Config, matcher_name: str) -> Tuple[str, Any, int]:

    for subtask_id, subtask_name in enumerate(config.get_subtasks()):
        scores = []
        runtimes = []
        
        ground_truth = config.get_ground_truth_set() # or get_ground_truth()
        source = config.get_source()
        target = config.get_target()
        
        if ground_truth is None or source is None or target is None:
            logger.error("ground_truth/source/target is None!")
            return

        matcher = get_matcher(matcher_name)

        
        n_jobs = config.get_n_jobs()
        for job_idx in range(n_jobs):
            # record start time
            start = time.time()

            # matches: "source" | "target"
            matches = matcher.matching(
                config.get_usecase_name(),
                subtask_name,
                source,
                target,
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

            save_json_results(matches, f"{matcher_name}__top{config.get_top_k()}__{config.get_usecase_name()}__{subtask_name.split('/')[-1]}__{job_idx}.json")

        
            mrr_score = compute_mean_ranking_reciprocal(
                matches, ground_truth
            )
    
            all_metrics = matches.get_metrics(ground_truth)
    
            one2one_metrics = matches.one_to_one().get_metrics(
                ground_truth
            )
    
            all_metrics["MRR"] = mrr_score
            for metrix_name, score in one2one_metrics.items():
                all_metrics[f"one2one_{metrix_name}"] = score
    
            # scores.append(score)
            runtimes.append((end - start) * 1e3)

        yield subtask_name, all_metrics, int(sum(runtimes) / n_jobs)
