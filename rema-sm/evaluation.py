# Description: This file contains the evaluation functions.
from valentine.algorithms.matcher_results import MatcherResults


def convert_to_valentine_format(matched_columns, source_table, target_table):
    valentine_format = {}
    for source_column, matches in matched_columns.items():
        for target_column, score in matches:
            key = (source_table, source_column), (target_table, target_column)
            valentine_format[key] = score
    if isinstance(valentine_format, MatcherResults):
        return valentine_format
    return MatcherResults(valentine_format)


def evaluate_matches(matches, ground_truth):
    mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)

    all_metrics = matches.get_metrics(ground_truth)

    one2one_metrics = matches.one_to_one().get_metrics(ground_truth)

    all_metrics["MRR"] = mrr_score
    for metrix_name, score in one2one_metrics.items():
        all_metrics[f"one2one_{metrix_name}"] = score
        
    all_metrics["RecallAtK"] = calculate_recall_at_k(matches, ground_truth)

    return all_metrics


def calculate_recall_at_k(matches, ground_truth):
    ground_truth_set = set(frozenset(pair) for pair in ground_truth)
    correct_matches = 0
    for ((_, source_col), (_, target_col)), _ in matches.items():
        match_pair = frozenset((source_col, target_col))
        if match_pair in ground_truth_set:
            correct_matches += 1
            ground_truth_set.remove(match_pair)

    total_ground_truth = len(ground_truth)
    recall = correct_matches / total_ground_truth if total_ground_truth > 0 else 0

    return recall


def sort_matches(matches):
    sorted_matches = {entry[0][1]: [] for entry in matches}
    for entry in matches:
        sorted_matches[entry[0][1]].append((entry[1][1], matches[entry]))
    return sorted_matches


def compute_mean_ranking_reciprocal(matches, ground_truth):
    ordered_matches = sort_matches(matches)
    total_score = 0
    for input_col, target_col in ground_truth:
        score = 0
        # print("Input Col: ", input_col)
        if input_col in ordered_matches:
            ordered_matches_list = [v[0] for v in ordered_matches[input_col]]
            # position = -1
            if target_col in ordered_matches_list:
                position = ordered_matches_list.index(target_col)
                score = 1 / (position + 1)
            else:
                print(f"1- Mapping {input_col} -> {target_col} not found")
                for entry in ordered_matches[input_col]:
                    print(entry)
        total_score += score

    final_score = total_score / len(ground_truth)
    return final_score
