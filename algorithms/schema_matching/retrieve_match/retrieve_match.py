import argparse
import json
import os
import time
from typing import Dict, Tuple

import pandas as pd
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable

from .evaluation import convert_to_valentine_format, evaluate_matches
from .matcher import ColumnMatcher
from .retriever import ColumnRetriever
from .utils import default_converter, get_dataset_paths, get_samples, process_tables


class RetrieveMatch(BaseMatcher):
    def __init__(self, model_type, dataset, serialization, llm_model):
        self.retriever = ColumnRetriever(
            model_type=model_type, dataset=dataset, serialization=serialization
        )
        self.matcher = ColumnMatcher(llm_model=llm_model)

    def match(self, source_table, target_table, top_k, cand_k):
        orig_source_table, orig_target_table = source_table, target_table
        source_table = source_table.get_df()
        target_table = target_table.get_df()

        # print(orig_source_table.name,
        #     orig_target_table.name)

        start_time = time.time()
        source_values = {
            col: get_samples(source_table[col]) for col in source_table.columns
        }
        target_values = {
            col: get_samples(target_table[col]) for col in target_table.columns
        }
        matched_columns = self.retriever.find_matches(
            source_table, target_table, source_values, target_values, top_k
        )
        # print("Matched Columns:", matched_columns)

        if cand_k > 1:
            matched_columns = self.matcher.rematch(
                source_table,
                target_table,
                source_values,
                target_values,
                top_k,
                matched_columns,
                cand_k,
            )
            # print("Refined Matches:", matched_columns)
        runtime = time.time() - start_time

        converted_matches = convert_to_valentine_format(
            matched_columns,
            orig_source_table.name,
            orig_target_table.name,
        )
        return converted_matches, runtime, matched_columns

    def get_matches(
        self, source_table: BaseTable, target_table: BaseTable
    ) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        converted_matches, runtime, matched_columns = self.match(
            source_table, target_table, 20, 20
        )
        return converted_matches


def run_retrieve_match(args):
    source_tables_path, target_tables_path, gt_path = get_dataset_paths(args.dataset)
    rema = RetrieveMatch(
        args.model_type, args.dataset, args.serialization, args.llm_model
    )

    params = (
        f"{args.model_type}_{args.serialization}_{args.top_k}_{args.cand_k}_{args.llm_model}"
        if args.cand_k > 1 and args.llm_model != "gpt-4-turbo-preview"
        else f"{args.model_type}_{args.serialization}_{args.top_k}_{args.cand_k}"
    )
    target_dir = f"{args.dataset}/{params}"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    gt_df = pd.read_csv(gt_path)
    columns = [
        "usecase",
        "tablename",
        "top_k",
        "runtime",
        "Precision",
        "F1Score",
        "Recall",
        "PrecisionTop10Percent",
        "RecallAtSizeofGroundTruth",
        "MRR",
        "one2one_Precision",
        "one2one_F1Score",
        "one2one_Recall",
        "one2one_PrecisionTop10Percent",
        "one2one_RecallAtSizeofGroundTruth",
    ]

    if args.dataset not in ["gdc"]:
        results = []
        for source_path in os.listdir(source_tables_path):
            matches_filename = f"{target_dir}/{source_path.split('.')[0]}_matches.json"

            if os.path.exists(matches_filename):
                continue

            matches, runtime, orig_matches = rema.match(
                source_tables_path,
                target_tables_path,
                source_path,
                args.top_k,
                args.cand_k,
            )
            gt_rows = gt_df[gt_df["source_tab"] == source_path.split(".")[0]]
            ground_truth = [
                (row["source_col"], row["target_col"]) for _, row in gt_rows.iterrows()
            ]

            print("Ground Truth:", ground_truth)
            metrics = evaluate_matches(matches, ground_truth)
            print("Metrics:", metrics)
            # exit()

            metrics.update(
                {
                    "usecase": args.dataset,
                    "tablename": source_path.split(".")[0],
                    "top_k": args.top_k,
                    "runtime": runtime * 1000,
                }
            )

            results.append(metrics)

            with open(matches_filename, "w") as f:
                json.dump(orig_matches, f, indent=4, default=default_converter)

        all_metrics = pd.DataFrame(results, columns=columns, index=None)

        mertics_filename = f"{target_dir}/metrics.csv"
        if os.path.exists(mertics_filename):
            all_metrics_df = pd.read_csv(mertics_filename)
            all_metrics = pd.concat([all_metrics_df, all_metrics], ignore_index=True)
            all_metrics.to_csv(mertics_filename, index=False)
        else:
            all_metrics.to_csv(mertics_filename, index=False)

        avg_metrics = all_metrics.mean(numeric_only=True)
        print("Average Metrics:", avg_metrics)
        avg_metrics_filename = f"{target_dir}/avg_metrics.csv"
        avg_metrics_df = avg_metrics.to_frame().T
        avg_metrics_df.to_csv(avg_metrics_filename, mode="a", index=False)

        data, usecase = args.dataset.split("-")
        avg_metrics_df.insert(0, "usecase", usecase)
        all_avg_metrics_filename = f"{data}-all/{params}.csv"
        if os.path.exists(all_avg_metrics_filename):
            all_avg_metrics_df = pd.read_csv(all_avg_metrics_filename)
            all_avg_metrics_df = pd.concat(
                [all_avg_metrics_df, avg_metrics_df], ignore_index=True
            )
            all_avg_metrics_df.to_csv(all_avg_metrics_filename, index=False)
        else:
            if not os.path.exists(f"{data}-all"):
                os.makedirs(f"{data}-all")
            avg_metrics_df.to_csv(all_avg_metrics_filename, index=False)


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
        default="header_values_prefix",
        help="Column serialization method (header, header_values_default, header_values_prefix, header_values_repeat)",
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of top matches to return"
    )
    parser.add_argument(
        "--cand_k",
        type=int,
        default=20,
        help="Number of candidate matches to refine",
    )
    parser.add_argument(
        "--llm_model",
        default="gpt-4-turbo-preview",
        help="Type of LLM-based matcher (gpt-4-turbo-preview or gemma2:9b)",
    )

    args = parser.parse_args()
    run_retrieve_match(args)


if __name__ == "__main__":
    main()
