import os
import sys
import pandas as pd
import time
import datetime
import pprint
pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)
from valentine import valentine_match



project_path = os.getcwd()
sys.path.append(os.path.join(project_path))


from experiments.benchmarks.utils import compute_mean_ranking_reciprocal, create_result_file, record_result
import algorithms.schema_matching.match_maker.match_maker as mm
import algorithms.schema_matching.gpt_matcher.gpt_matcher as gpt_matcher

def get_gpt_method(method):
    if method == 'GPTMatcher':
        return gpt_matcher.GPTMatcher()
    elif method == 'GPTMatcherExample':
        return gpt_matcher.GPTMatcher(include_example=True)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_match_maker_matcher(method):
    if method=="MatchMaker":
        return mm.MatchMaker()
    elif method=="MatchMakerBP":
        return mm.MatchMaker(use_bp_reranker=True)
    elif method.startswith('MatchMakerGPT'):
        print("Method: ", method)
        topk = int(method.split('_')[1])
        return mm.MatchMaker(use_gpt_reranker=True, topk=topk)

def get_matcher(method):
    if method.startswith('GPT'):
        return get_gpt_method(method)
    elif method.startswith('MatchMaker'):
        return get_match_maker_matcher(method)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_ablation(BENCHMARK='gdc_studies', DATASET='gdc_studies', ROOT='data/gdc'):

    HEADER = ['benchmark', 'dataset', 'source_table', 'target_table', 'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches', 'method', 'runtime', 'mrr',  'All_Precision', 'All_F1Score', 'All_Recall', 'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
              'One2One_Precision', 'One2One_F1Score', 'One2One_Recall', 'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth']

    # Create results directory and file
    results_dir = os.path.join(
        project_path, 'results', 'ablations', 'gpt_reranker',
        BENCHMARK, DATASET
    )
    result_file = os.path.join(
        results_dir,
        f'{BENCHMARK}_{DATASET}_gpt_reranker_results_{
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    )
    create_result_file(results_dir, result_file, HEADER)

    target_file = os.path.join(
        ROOT, 'target-tables', 'gdc_unique_columns_concat_values.csv')

    df_target = pd.read_csv(target_file, low_memory=False)

    studies_path = os.path.join(ROOT, 'source-tables')
    gt_path = os.path.join(ROOT, 'ground-truth')

    for gt_file in os.listdir(gt_path):
        if gt_file.endswith('.csv'):

            print(f"Processing {gt_file}")

            if gt_file != 'Huang.csv':
                continue

            source_file = os.path.join(studies_path, gt_file)
            df_source = pd.read_csv(source_file)

            gt_df = pd.read_csv(os.path.join(gt_path, gt_file))
            gt_df.dropna(inplace=True)
            ground_truth = list(gt_df.itertuples(index=False, name=None))

            matchers = ["MatchMaker",  "MatchMakerFT"]
            # matchers = [ "MatchMaker", "MatchMakerBP", "MatchMakerGPT_10", "MatchMakerGPT_20", "GPTMatcher", "GPTMatcherExample"]
            matchers = [ "GPTMatcher", "GPTMatcherExample"]

            for matcher in matchers:
                # print(f"Matcher: {matcher}, Source: {source_file}, Target: {target_file}")

                method_name = matcher
                matcher = get_matcher(matcher)

                start_time = time.time()
                matches = valentine_match(df_source, df_target, matcher)
                end_time = time.time()
                runtime = end_time - start_time

                mrr_score = compute_mean_ranking_reciprocal(
                    matches, ground_truth)

                all_metrics = matches.get_metrics(ground_truth)

                recallAtGT = all_metrics['RecallAtSizeofGroundTruth']

                print('File: ', gt_file, ' and ', method_name, " with MRR Score: ",
                      mrr_score, ", RecallAtGT: ", recallAtGT, " and Runtime: ", runtime)

                matches = matches.one_to_one()
                one2one_metrics = matches.get_metrics(ground_truth)

                ncols_src = str(df_source.shape[1])
                ncols_tgt = str(df_target.shape[1])
                nrows_src = str(df_source.shape[0])
                nrows_tgt = str(df_target.shape[0])

                nmatches = len(ground_truth)

                result = [BENCHMARK, DATASET, 'gdc_table', gt_file, ncols_src, ncols_tgt, nrows_src, nrows_tgt, nmatches, method_name, runtime, mrr_score, all_metrics['Precision'], all_metrics['F1Score'], all_metrics['Recall'], all_metrics['PrecisionTop10Percent'], all_metrics['RecallAtSizeofGroundTruth'],
                          one2one_metrics['Precision'], one2one_metrics['F1Score'], one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'], one2one_metrics['RecallAtSizeofGroundTruth']]

                record_result(result_file, result)
            print("\n")


if __name__ == '__main__':
    run_ablation()
