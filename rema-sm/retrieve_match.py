import argparse
import numpy as np
import pandas as pd
import os
import time
import json

from retriever import ColumnRetriever
from matcher import ColumnMatcher
from utils import get_dataset_paths, process_tables, get_samples, default_converter
from evaluation import evaluate_matches, convert_to_valentine_format


class RetrieveMatch:
    def __init__(self, model_type, dataset, serialization, llm_model, norm):
        self.retriever = ColumnRetriever(
            model_type=model_type,
            dataset=dataset,
            serialization=serialization,
            norm=norm,
        )
        self.matcher = ColumnMatcher(llm_model=llm_model)
        
    def _min_max_normalize(self, matched_columns):
        normalized_columns = {}
        for key, values in matched_columns.items():
            # Extract only the weights for min-max normalization
            weights = [weight for _, weight in values]
            min_w = min(weights)
            max_w = max(weights)
            range_w = max_w - min_w
            normalized_list = [(name, (weight - min_w) / range_w if range_w > 0 else 0) for name, weight in values]
            normalized_columns[key] = normalized_list
        return normalized_columns

    def _identify_low_confidence_sources(self, matched_columns, threshold=75):
        variances = []

        for s_col, matches in matched_columns.items():
            if matches:
                _, scores = zip(*matches)
                variances.append(np.var(scores))

        # Calculate thresholds based on variances
        if variances:
            variance_threshold = np.percentile(
                variances, threshold
            )  # Upper 75th percentile for variance

            unconf_matched_columns = {}
            conf_matched_columns = {}

            for s_col, matches in matched_columns.items():
                _, scores = zip(*matches)
                variance = np.var(scores)

                if variance < variance_threshold:
                    unconf_matched_columns[s_col] = matches
                else:
                    conf_matched_columns[s_col] = matches

            print("Variance Threshold:", variance_threshold)
            return unconf_matched_columns, conf_matched_columns
        else:
            return {}, matched_columns

    def match(self, source_tables_path, target_tables_path, source_path, args):
        top_k, cand_k, conf_prune = args.top_k, args.cand_k, args.conf_prune
        source_table = pd.read_csv(os.path.join(source_tables_path, source_path))
        target_path = source_path.replace("source", "target")
        target_table = pd.read_csv(os.path.join(target_tables_path, target_path))
        source_table, target_table = process_tables(source_table, target_table)

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
            columns_to_refine, matched_columns = (
                self._identify_low_confidence_sources(matched_columns)
                if conf_prune
                else (matched_columns, {})
            )
            unconf_size = len(columns_to_refine)
            print("Number of columns to refine:", unconf_size)
            if unconf_size > 0:
                refined_columns = self.matcher.rematch(
                    source_table,
                    target_table,
                    source_values,
                    target_values,
                    top_k,
                    columns_to_refine,
                    cand_k,
                )
                print("Refined Matches:", refined_columns)
                matched_columns.update(refined_columns)
        print("Matched Columns:", matched_columns)
        normzlized_matches = self._min_max_normalize(matched_columns)
        print("Normalized Matches:", normzlized_matches)
        exit()
        runtime = time.time() - start_time

        converted_matches = convert_to_valentine_format(
            normzlized_matches,
            source_path.replace(".csv", ""),
            target_path.replace(".csv", ""),
        )
        return converted_matches, runtime, matched_columns


def run_retrieve_match(args):
    source_tables_path, target_tables_path, gt_path = get_dataset_paths(args.dataset)
    norm = True if args.cand_k == 1 or args.conf_prune else False
    print("Normalization: ", norm)
    rema = RetrieveMatch(
        args.model_type, args.dataset, args.serialization, args.llm_model, norm
    )

    params = f"{args.model_type}_{args.serialization}_{args.top_k}_{args.cand_k}"
    if args.cand_k > 1:
        params += f"_{args.llm_model}"
    if args.conf_prune:
        params += "_conf_prune"

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
                source_tables_path, target_tables_path, source_path, args
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
        "--cand_k", type=int, default=20, help="Number of candidate matches to refine",
    )
    parser.add_argument(
        "--llm_model",
        default="gpt-4o-mini",
        help="Type of LLM-based matcher (gpt-4-turbo-preview, gpt-4o-mini, gemma2:9b, llama3.1:8b-instruct-fp16)",
    )
    parser.add_argument(
        "--conf_prune",
        action="store_true",
        help="Only refine matches with low confidence",
    )

    args = parser.parse_args()
    run_retrieve_match(args)


if __name__ == "__main__":
    main()
