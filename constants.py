import os

INDEX_DIR = "index_files"
PARTIAL_DUMP_THRESHOLD = 10000

# Weights for important words (spec: bold, h1-h3, title treated as more important)
TITLE_WEIGHT = 3.0
HEADING_WEIGHT = 2.0
BOLD_WEIGHT = 2.0

os.makedirs(INDEX_DIR, exist_ok=True)

MAPPING_FILE   = os.path.join(INDEX_DIR, "url_mappings.pkl")
POSTINGS_FILE  = os.path.join(INDEX_DIR, "postings.bin")
TERM_DICT_FILE = os.path.join(INDEX_DIR, "term_dict.pkl")