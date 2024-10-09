# ***************************************************************************************
schema_matching_data = {
    "amazon_google_exp": ["train", "../../datasets/amazon_google_exp/", "recall"],
    "beeradvo_ratebeer": ["train", "../../datasets/beeradvo_ratebeer/", "recall"],
    "dblp_acm": ["train", "../../datasets/dblp_acm/", "recall"],
    "dblp_scholar": ["train", "../../datasets/dblp_scholar/", "recall"],
    "fodors_zagats": ["train", "../../datasets/fodors_zagats/", "recall"],
    "itunes_amazon": ["train", "../../datasets/itunes_amazon/", "recall"],
    "walmart_amazon": ["train", "../../datasets/walmart_amazon/", "recall"],
    # "cao": ["train", "../../datasets/cao/", "recall"],
    # "clark": ["train", "../../datasets/clark/", "recall"],
    # "dou": ["train", "../../datasets/dou/", "recall"],
    # "huang": ["train", "../../datasets/huang/", "recall"],
    # "krug": ["train", "../../datasets/krug/", "recall"],
    # "satpathy": ["test", "../../datasets/satpathy/", "recall"],
    # "vasaikar": ["test", "../../datasets/vasaikar/", "recall"],
    # "wang": ["test", "../../datasets/wang/", "recall"],
}


# 2:1:7 split gold
entity_alignment_data = {
    "dbp_yg": ["train", "data/dbp_yg/", "hit"],
    "dbp_wd": ["test", "data/dbp_wd/", "hit"],
}

# 2:1:7
string_matching_data = {
    "smurf1": ["train", "data/smurf-addr/", "f1"],
    "smurf2": ["train", "data/smurf-names/", "f1"],
    "smurf3": ["train", "data/smurf-res/", "f1"],
    "smurf4": ["test", "data/smurf-prod/", "f1"],
    "smurf5": ["train", "data/smurf-cit/", "f1"],
}

# 3:1:1 directly use
new_deepmatcher_data = {
    "m1": ["train", "data/em-wa/", "f1"],
    "m2": ["test", "data/em-ds/", "f1"],
    "m3": ["train", "data/em-fz/", "f1"],
    "m4": ["train", "data/em-ia/", "f1"],
    "m5": ["train", "data/em-beer/", "f1"],
}

# 2:1:7
new_schema_matching_data = {
    "fab": ["train", "data/fabricated_dataset/", "recall"],
    "deepmdatasets": ["test", "data/DeepMDatasets/", "recall"],
}

# 2:1:7
ontology_matching_data = {"cw": ["train", "data/Illinois-onm/", "acc"]}

# 2:1:7
column_type_data = {
    "efth": ["train", "data/efthymiou/", "acc"],
    "t2d": ["train", "data/t2d_col_type_anno/", "acc"],
    "limaya": ["test", "data/Limaye_col_type_anno/", "acc"],
}

# 2:1:7
entity_linking_data = {
    "t2d": ["train", "data/t2d/", "f1"],
    "limaya": ["test", "data/Limaye/", "f1"],
}
