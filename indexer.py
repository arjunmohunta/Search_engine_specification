#Team members id - 45885756, 87608468, 50527916, 80654131
import os
import json
import pickle
import re
import sys
import hashlib
from collections import defaultdict
from bs4 import BeautifulSoup
import warnings
from constants import INDEX_DIR, PARTIAL_DUMP_THRESHOLD, MAPPING_FILE, POSTINGS_FILE, TERM_DICT_FILE
from constants import TITLE_WEIGHT, HEADING_WEIGHT, BOLD_WEIGHT

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
except ModuleNotFoundError:
    _stemmer = None

def tokenize(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if _stemmer:
        tokens = [_stemmer.stem(t) for t in tokens]
    return tokens

def extract_text_regions(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        body_text = soup.get_text()
        title_text = soup.title.get_text() if soup.title and soup.title.string else ""
        heading_text = " ".join(t.get_text() for t in soup.find_all(["h1","h2","h3"]) if t.get_text())
        bold_text = " ".join(t.get_text() for t in soup.find_all(["b","strong"]) if t.get_text())
        return body_text, title_text, heading_text, bold_text
    except Exception:
        return "", "", "", ""

class Indexer:
    def __init__(self):
        self.index = {}
        self.mapping = {}
        self.doc_count = 0
        self.partial_index_count = 0
        self.seen_hashes = set()
        self.duplicate_count = 0

    def add_token(self, token, doc_id, weight=1.0):
        if token not in self.index:
            self.index[token] = {}
        if doc_id not in self.index[token]:
            self.index[token][doc_id] = {"tf": 0, "wt": 0.0}
        self.index[token][doc_id]["tf"] += 1
        self.index[token][doc_id]["wt"] += weight

    def add_document(self, doc_id, content):
        body, title, heading, bold = extract_text_regions(content)
        for token in tokenize(body):
            self.add_token(token, doc_id, 1.0)
        for token in tokenize(title):
            self.add_token(token, doc_id, TITLE_WEIGHT)
        for token in tokenize(heading):
            self.add_token(token, doc_id, HEADING_WEIGHT)
        for token in tokenize(bold):
            self.add_token(token, doc_id, BOLD_WEIGHT)

    def flush_partial_index(self):
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
                    continue
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                content = data.get("content", "")
                url     = data.get("url", "")
                if not content or not url:
                    continue

                # duplicate check
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if content_hash in self.seen_hashes:
                    self.duplicate_count += 1
                    continue
                self.seen_hashes.add(content_hash)

                doc_id = self.doc_count
                self.doc_count += 1
                self.mapping[doc_id] = url.split("#")[0]

                self.add_document(doc_id, content)

                if len(self.index) >= PARTIAL_DUMP_THRESHOLD:
                    self.flush_partial_index()

        if self.index:
            self.flush_partial_index()

        with open(MAPPING_FILE, "wb") as f:
            pickle.dump((self.mapping, self.doc_count), f)
        print(f"Saved mapping ({self.doc_count} docs, {self.duplicate_count} duplicates skipped)")

    def merge_partials(self):
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
                            merged[doc_id] = {"tf": 0, "wt": 0.0}
                        merged[doc_id]["tf"] += values["tf"]
                        merged[doc_id]["wt"] += values.get("wt", values["tf"])
                df = len(merged)
                offset = postings_f.tell()
                data = pickle.dumps(merged)
                postings_f.write(data)
                term_dict[term] = (offset, len(data), df)

        with open(TERM_DICT_FILE, "wb") as f:
            pickle.dump(term_dict, f)

        return term_dict

    def compute_analytics(self, term_dict):
        total_size = (
            os.path.getsize(POSTINGS_FILE) +
            os.path.getsize(TERM_DICT_FILE) +
            os.path.getsize(MAPPING_FILE)
        )
        print("\n===== INDEX ANALYTICS =====")
        print(f"Documents indexed:      {self.doc_count}")
        print(f"Unique tokens:          {len(term_dict)}")
        print(f"Index size (KB):        {round(total_size / 1024,2)}")
        print(f"Exact duplicates skipped:{self.duplicate_count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python indexer.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    indexer = Indexer()
    indexer.process_directory(dataset_path)
    term_dict = indexer.merge_partials()
    indexer.compute_analytics(term_dict)
