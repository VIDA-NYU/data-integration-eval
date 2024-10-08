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

    target = args.target
    if args.usecase:
        source = os.path.join(args.usecase, "source.csv")
        ground_truth = os.path.join(args.usecase, "groundtruth.csv")
        if os.path.exists(os.path.join(args.usecase, "target.csv")):
            target = os.path.join(args.usecase, "target.csv")
    else:
        source = args.source
        ground_truth = args.ground_truth

    config = Config(
        usecase=args.usecase,
        source=source,
        target=target,
        ground_truth=ground_truth,
        n_jobs=args.n_jobs,
        top_k=args.top_k,
    )

    all_metrics, runtime = matching(config, args.matcher)

    parse_results(source, target, args.matcher, runtime, all_metrics, args.output)


if __name__ == "__main__":
    main()
