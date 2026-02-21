#Team members id - 45885756, 87608468, 50527916, 80654131
import os
import json
import pickle
import re
import sys
import hashlib
from collections import defaultdict
from bs4 import BeautifulSoup
try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
except ModuleNotFoundError:
    _stemmer = None

INDEX_DIR = "index_files"
PARTIAL_DUMP_THRESHOLD = 10000  
os.makedirs(INDEX_DIR, exist_ok=True)

MAPPING_FILE   = os.path.join(INDEX_DIR, "url_mappings.pkl")
POSTINGS_FILE  = os.path.join(INDEX_DIR, "postings.bin")
TERM_DICT_FILE = os.path.join(INDEX_DIR, "term_dict.pkl")


def tokenize(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if _stemmer:
        tokens = [_stemmer.stem(t) for t in tokens]
    return tokens


def extract_text(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()
    except Exception:
        return ""


class Indexer:
    def __init__(self):
        self.index = {}            
        self.mapping = {}          
        self.doc_count = 0         
        self.partial_index_count = 0
        self.seen_hashes = set()   
        self.duplicate_count = 0

    def add_token(self, token, doc_id):
        if token not in self.index:
            self.index[token] = {}
        if doc_id not in self.index[token]:
            self.index[token][doc_id] = {"tf": 0}
        self.index[token][doc_id]["tf"] += 1

    def add_document(self, doc_id, content):
        text = extract_text(content)
        for token in tokenize(text):
            self.add_token(token, doc_id)

    def flush_partial_index(self):
        """Save current in-memory index as a partial file."""
        filename = os.path.join(INDEX_DIR, f"partial_{self.partial_index_count}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self.index, f)
        print(f"Flushed partial index {self.partial_index_count} with {len(self.index)} tokens")
        self.partial_index_count += 1
        self.index = {}

    def process_directory(self, root_dir):
        for root, _, files in os.walk(root_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue    # check for json file
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue    # skip bad json files

                content = data.get("content", "")
                url     = data.get("url", "")
                if not content or not url:
                    continue    # if url or content is empty do not continue

                # use hashes to check for duplicates
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if content_hash in self.seen_hashes:
                    self.duplicate_count += 1
                    continue
                self.seen_hashes.add(content_hash)

                doc_id = self.doc_count
                self.doc_count += 1
                self.mapping[doc_id] = url

                self.add_document(doc_id, content)

                if len(self.index) >= PARTIAL_DUMP_THRESHOLD:
                    self.flush_partial_index()

        if self.index:
            self.flush_partial_index()

        
        with open(MAPPING_FILE, "wb") as f:
            pickle.dump(self.mapping, f)
        print(f"URL mapping saved ({self.doc_count} documents, {self.duplicate_count} duplicates skipped).")

    def merge_partials(self):
        """Merge all partial indices into final postings file."""
        partials = {}
        term_to_partials = defaultdict(list)

        for i in range(self.partial_index_count):
            filename = os.path.join(INDEX_DIR, f"partial_{i}.pkl")
            with open(filename, "rb") as f:
                partials[i] = pickle.load(f)
            for term in partials[i]:
                term_to_partials[term].append(i)
            
        term_dict = {}

        with open(POSTINGS_FILE, "wb") as postings_f:
            for term in sorted(term_to_partials):
                merged = {}
                for i in term_to_partials[term]:
                    for doc_id, values in partials[i][term].items():
                        if doc_id not in merged:
                            merged[doc_id] = {"tf": 0}
                        merged[doc_id]["tf"] += values["tf"]

                offset = postings_f.tell()
                data   = pickle.dumps(merged)
                postings_f.write(data)
                term_dict[term] = (offset, len(data))

                
        with open(TERM_DICT_FILE, "wb") as f:
            pickle.dump(term_dict, f)
        

        return term_dict

    def compute_analytics(self, term_dict):
        total_size_bytes = (
            os.path.getsize(POSTINGS_FILE) +
            os.path.getsize(TERM_DICT_FILE) +
            os.path.getsize(MAPPING_FILE)
        )
        print("\n===== INDEX ANALYTICS =====")
        print(f"Number of indexed documents: {self.doc_count}")
        print(f"Number of unique tokens:     {len(term_dict)}")
        print(f"Total index size (KB):       {round(total_size_bytes / 1024, 2)}")
        print(f"Exact duplicates skipped:    {self.duplicate_count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python indexer.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    indexer = Indexer()
    indexer.process_directory(dataset_path)
    term_dict = indexer.merge_partials()
    indexer.compute_analytics(term_dict)
