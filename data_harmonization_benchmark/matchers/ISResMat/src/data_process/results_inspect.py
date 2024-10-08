'''
This script mainly identifies which columns the current algorithm did not perform well on.
'''

import ast
import json
# execution outcomes
res_json = '../../../data/output/inst-001/Semantically-Joinable-assays_both_50_50_ac4_av.json'
# expected outcomes
exp_json = '../../../data/sm-valentine-m1/assays/Semantically-Joinable/assays_both_50_50_ac4_av/assays_both_50_50_ac4_av_mapping.json'

with open(res_json, 'r') as jfile:
    data = json.load(jfile)

res_match_dc = data['matches']
res_match_dc = {ast.literal_eval(key): value for key, value in res_match_dc.items()}
# print(res_match_dc)


expected_matches = set()
with open(exp_json) as json_file:
    mappings: list = json.load(json_file)["matches"]
    for mapping in mappings:
        expected_matches.add(
            frozenset(((mapping["source_table"], mapping["source_column"]),
                       (mapping["target_table"], mapping["target_column"]))))

res_matches = list(map(lambda m: frozenset(m), list(res_match_dc.keys())))

tp = 0
fn = 0
for exp_match in expected_matches:
    if exp_match in res_matches[:len(expected_matches)]:
        tp = tp + 1
    else:
        fn = fn + 1
        print(exp_match)
print(tp, fn)
