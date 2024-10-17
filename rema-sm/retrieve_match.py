import argparse
import pandas as pd
import os

from retriever import ColumnRetriever
from matcher import ColumnMatcher
from utils import get_dataset_paths, process_tables
from evaluation import evaluate_matches, convert_to_valentine_format

API_KEY = "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"


class RetrieveMatch:
    def __init__(self, model_type, dataset, serialization):
        self.retriever = ColumnRetriever(
            model_type=model_type, dataset=dataset, serialization=serialization
        )
        self.matcher = ColumnMatcher(api_key=API_KEY)

    def match(self, source_tables_path, target_tables_path, source_path, top_k):
        source_table = pd.read_csv(os.path.join(source_tables_path, source_path))
        target_path = source_path.replace("source", "target")
        target_table = pd.read_csv(os.path.join(target_tables_path, target_path))
        source_table, target_table = process_tables(source_table, target_table)

        matched_columns = self.retriever.find_matches(source_table, target_table, top_k)
        print("Matched Columns:", matched_columns)
        refined_matches = self.matcher.rematch(source_table, matched_columns, top_k)
        print("Refined Matches:", refined_matches)
        converted_matches = convert_to_valentine_format(
            matched_columns,
            source_path.replace(".csv", ""),
            target_path.replace(".csv", ""),
        )
        print("Converted Matches:", converted_matches)
        return converted_matches


def run_retrieve_match(args):
    source_tables_path, target_tables_path, gt_path = get_dataset_paths(args.dataset)
    rema = RetrieveMatch(args.model_type, args.dataset, args.serialization)

    gt_df = pd.read_csv(gt_path)

    if args.dataset not in ["gdc"]:
        for source_path in os.listdir(source_tables_path):
            matches = rema.match(
                source_tables_path, target_tables_path, source_path, args.top_k
            )
            gt_rows = gt_df[gt_df["source_tab"] == source_path.split(".")[0]]
            ground_truth = []
            for _, row in gt_rows.iterrows():
                ground_truth.append((row["source_col"], row["target_col"]))
            print("Ground Truth:", ground_truth)
            metrics = evaluate_matches(matches, ground_truth)
            print("Metrics:", metrics)
            exit()


def main():
    parser = argparse.ArgumentParser(
        description="Match columns between source and target tables using pretrained models."
    )
    parser.add_argument(
        "--dataset",
        default="chembl-joinable",
        help="Name of the dataset for model customization",
    )
    parser.add_argument(
        "--model_type",
        default="roberta-zs",
        help="Type of model (roberta-zs, roberta-ft, mpnet-zs, mpnet-ft, arctic-zs, arctic-ft)",
    )
    parser.add_argument(
        "--serialization",
        default="header_values",
        help="Column serialization method (header, header_values)",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top matches to return"
    )

    args = parser.parse_args()
    run_retrieve_match(args)


if __name__ == "__main__":
    main()
