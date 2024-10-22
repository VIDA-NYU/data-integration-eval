import argparse
import logging
import os

import pandas as pd

from config import Config
from matching import matching
from utils.result_proc import parse_results

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", "--usecase", type=str, help="Usecase")
    parser.add_argument(
        "-m", "--matcher", type=str, help="Matcher to use", default="jaccard_distance"
    )
    parser.add_argument("-s", "--source", type=str, help="Source file path")
    parser.add_argument(
        "-t", "--target", type=str, help="Target file path", default="gdc"
    )
    parser.add_argument("-g", "--ground-truth", type=str, help="Ground truth file path")
    parser.add_argument(
        "-o", "--output", type=str, help="Output file path", default="results.csv"
    )
    parser.add_argument("-n", "--n-jobs", type=int, help="Number of jobruns", default=1)
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        help="Top k results for matching candidates",
        default=10,
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Use case: %s", args.usecase)
    logger.info("Running benchmark with matcher: %s", args.matcher)
    logger.info("Source file: %s", args.source)
    logger.info("Target file: %s", args.target)
    logger.info("Ground truth file: %s", args.ground_truth)
    logger.info("Output file: %s", args.output)
    logger.info("Number of runs: %d", args.n_jobs)
    logger.info("Top k candidates: %d", args.top_k)

    sources = []
    targets = []
    ground_truths = []
    subtasks = []

    def init_data(path):
        if os.path.exists(os.path.join(path, "source.csv")) and os.path.exists(os.path.join(path, "target.csv")) and os.path.exists(os.path.join(path, "groundtruth.csv")):
            sources.append(os.path.join(path, "source.csv"))
            targets.append(os.path.join(path, "target.csv"))
            ground_truths.append(os.path.join(path, "groundtruth.csv"))
            subtasks.append(path)
        for root, dirs, _ in os.walk(path):
            for dir in dirs:
                abs_path = os.path.join(root, dir)
                init_data(abs_path)
    
    if args.usecase:
        init_data(args.usecase)
    else:
        sources = [args.source]
        targets = [args.target]
        ground_truths = [args.ground_truth]
        subtasks = [args.usecase]

    config = Config(
        usecase=args.usecase,
        subtasks=subtasks,
        sources=sources,
        targets=targets,
        ground_truths=ground_truths,
        n_jobs=args.n_jobs,
        top_k=args.top_k,
    )

    for subtask_name, all_metrics, runtime in matching(config, args.matcher):
        parse_results(args.usecase, subtask_name, args.matcher, args.top_k, runtime, all_metrics, args.output)


if __name__ == "__main__":
    main()
