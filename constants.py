import os

INDEX_DIR = "index_files"
PARTIAL_DUMP_THRESHOLD = 10000

os.makedirs(INDEX_DIR, exist_ok=True)

MAPPING_FILE   = os.path.join(INDEX_DIR, "url_mappings.pkl")
POSTINGS_FILE  = os.path.join(INDEX_DIR, "postings.bin")
TERM_DICT_FILE = os.path.join(INDEX_DIR, "term_dict.pkl")