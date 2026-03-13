import os

INDEX_DIR = "index_files"
PARTIAL_DUMP_THRESHOLD = 10000
TOP_K_RESULTS  = 10

TITLE_WEIGHT = 3.0
HEADING_WEIGHT = 2.0
BOLD_WEIGHT = 2.0
ANCHOR_WEIGHT = 1.5

os.makedirs(INDEX_DIR, exist_ok=True)

MAPPING_FILE   = os.path.join(INDEX_DIR, "url_mappings.pkl")
POSTINGS_FILE  = os.path.join(INDEX_DIR, "postings.bin")
TERM_DICT_FILE = os.path.join(INDEX_DIR, "term_dict.pkl")
PAGERANK_FILE  = os.path.join(INDEX_DIR, "pagerank.pkl")

# Near-duplicate detection (SimHash)
SIMHASH_BITS = 64
SIMHASH_HAMMING_THRESHOLD = 3

# PageRank
PAGERANK_DAMPING = 0.85
PAGERANK_ITERATIONS = 20
PAGERANK_ALPHA = 0.3  # blend weight in final score
