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
