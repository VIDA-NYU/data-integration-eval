import os

# root dir
from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent)
#################################################################################
# model dir
BERT_BASE_UNCASED_DIR = os.path.join(ROOT_DIR, 'resources/bert-base-uncased')