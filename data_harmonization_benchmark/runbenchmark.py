import argparse
import logging
import os

import pandas as pd

from config import Config
from matching import matching
from utils.result_proc import save_result_csv

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
    parser.add_argument("-n", "--njobs", type=int, help="Number of jobruns", default=1)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Use case: %s", args.usecase)
    logger.info("Running benchmark with matcher: %s", args.matcher)
    logger.info("Source file: %s", args.source)
    logger.info("Target file: %s", args.target)
    logger.info("Ground truth file: %s", args.ground_truth)
    logger.info("Output file: %s", args.output)
    logger.info("Number of runs: %d", args.njobs)

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
        source=source, target=target, ground_truth=ground_truth, n_jobs=args.njobs
    )

    score, runtime = matching(config, args.matcher)

    result_df = pd.DataFrame(
        {
            "source": [source],
            "target": [target],
            "matcher": [args.matcher],
            "score": [score],
            "runtime": [runtime],
        }
    )

    save_result_csv(result_df, args.output, append=True)


if __name__ == "__main__":
    main()
