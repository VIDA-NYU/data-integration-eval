run_isresmat() {
  python -m isresmat \
    --n-trn-cols=200 \
    --batch-size=1 \
    --frag-height=6 \
    --frag-width=12 \
    --learning-rate=3e-5 \
    --col-name-prob=0 \
    --store-matches=0 \
    --comment=inst-001 \
    --dataset-name=$data_name \
    --orig-file-src=$src_orig_file \
    --orig-file-tgt=$tgt_orig_file \
    --orig-file-golden-matches=$golden_mappings \
    --process-mode=1 \
    --n-val-cols=1
}

data_dir="../../datasets/amazon_google_exp/"
data_name=amazon_google_exp
src_orig_file="${data_dir}/source.csv"
tgt_orig_file="${data_dir}/target.csv"
golden_mappings="${data_dir}/groundtruth.csv"

run_isresmat