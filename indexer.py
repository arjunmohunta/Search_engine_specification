#Team members id - 45885756, 87608468, 50527916, 80654131
import os
import json
import pickle
import re
import sys
import hashlib
from collections import defaultdict
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import warnings
from constants import (INDEX_DIR, PARTIAL_DUMP_THRESHOLD, MAPPING_FILE,
                       POSTINGS_FILE, TERM_DICT_FILE, PAGERANK_FILE,
                       TITLE_WEIGHT, HEADING_WEIGHT, BOLD_WEIGHT,
                       SIMHASH_BITS, SIMHASH_HAMMING_THRESHOLD,
                       PAGERANK_DAMPING, PAGERANK_ITERATIONS)

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
try:
    from bs4 import XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
except ImportError:
    pass

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

def parse_document(html, base_url=""):
    """Parse HTML into text regions and outgoing links (single pass)."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        body_text = soup.get_text()
        title_text = soup.title.get_text() if soup.title and soup.title.string else ""
        heading_text = " ".join(t.get_text() for t in soup.find_all(["h1", "h2", "h3"]) if t.get_text())
        bold_text = " ".join(t.get_text() for t in soup.find_all(["b", "strong"]) if t.get_text())
        links = []
        if base_url:
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                if not href or href.startswith(("javascript:", "mailto:", "#")):
                    continue
                full_url = urljoin(base_url, href).split("#")[0]
                links.append(full_url)
        return body_text, title_text, heading_text, bold_text, links
    except Exception:
        return "", "", "", "", []

# Compatibility alias for any external callers
def extract_text_regions(html):
    body, title, heading, bold, _ = parse_document(html)
    return body, title, heading, bold


# ---------------------------------------------------------------------------
# SimHash near-duplicate detection
# ---------------------------------------------------------------------------

def simhash(tokens, num_bits=SIMHASH_BITS):
    """Compute a SimHash fingerprint for a list of tokens."""
    v = [0] * num_bits
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest(), 16)
        for i in range(num_bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(num_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint

def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class Indexer:
    def __init__(self):
        self.index = {}
        self.mapping = {}
        self.doc_count = 0
        self.partial_index_count = 0
        self.seen_hashes = set()
        self.seen_simhashes = [] # list of SimHash fingerprints
        self.duplicate_count = 0
        self.near_duplicate_count = 0
        self.raw_links = {} # doc_id -> [outgoing urls]

    def add_token(self, token, doc_id, weight=1.0, position=None):
        if token not in self.index:
            self.index[token] = {}
        if doc_id not in self.index[token]:
            self.index[token][doc_id] = {"tf": 0, "wt": 0.0, "pos": []}
        self.index[token][doc_id]["tf"] += 1
        self.index[token][doc_id]["wt"] += weight
        if position is not None:
            self.index[token][doc_id]["pos"].append(position)

    def add_document(self, doc_id, body, title, heading, bold):
        """Index pre-parsed text regions, storing body token positions."""
        body_tokens = tokenize(body)
        for pos, token in enumerate(body_tokens):
            self.add_token(token, doc_id, 1.0, position=pos)
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

                # Exact duplicate check (SHA-256)
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if content_hash in self.seen_hashes:
                    self.duplicate_count += 1
                    continue
                self.seen_hashes.add(content_hash)

                # Parse HTML once
                body, title, heading, bold, links = parse_document(content, url)

                # Near-duplicate check (SimHash)
                # Cap body length for speed; include title for distinctiveness
                sh = simhash(tokenize(body[:5000] + " " + title))
                if any(hamming_distance(sh, seen) <= SIMHASH_HAMMING_THRESHOLD
                       for seen in self.seen_simhashes):
                    self.near_duplicate_count += 1
                    continue
                self.seen_simhashes.append(sh)

                doc_id = self.doc_count
                self.doc_count += 1
                self.mapping[doc_id] = url.split("#")[0]
                self.raw_links[doc_id] = links

                self.add_document(doc_id, body, title, heading, bold)

                if len(self.index) >= PARTIAL_DUMP_THRESHOLD:
                    self.flush_partial_index()

        if self.index:
            self.flush_partial_index()

        with open(MAPPING_FILE, "wb") as f:
            pickle.dump((self.mapping, self.doc_count), f)
        print(f"Saved mapping ({self.doc_count} docs, "
              f"{self.duplicate_count} exact + {self.near_duplicate_count} near-duplicates skipped)")

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
                            merged[doc_id] = {"tf": 0, "wt": 0.0, "pos": []}
                        merged[doc_id]["tf"] += values["tf"]
                        merged[doc_id]["wt"] += values.get("wt", values["tf"])
                        if "pos" in values:
                            merged[doc_id]["pos"].extend(values["pos"])
                df = len(merged)
                offset = postings_f.tell()
                data = pickle.dumps(merged)
                postings_f.write(data)
                term_dict[term] = (offset, len(data), df)

        with open(TERM_DICT_FILE, "wb") as f:
            pickle.dump(term_dict, f)

        return term_dict

    # -----------------------------------------------------------------------
    # PageRank
    # -----------------------------------------------------------------------

    def compute_pagerank(self):
        """Build in-corpus link graph, run PageRank, save scores."""
        url_to_docid = {url: doc_id for doc_id, url in self.mapping.items()}
        N = self.doc_count
        if N == 0:
            return {}

        # Build adjacency lists (within-corpus links only)
        out_links = defaultdict(set)
        in_links = defaultdict(set)
        for doc_id, urls in self.raw_links.items():
            for url in urls:
                target = url_to_docid.get(url)
                if target is not None and target != doc_id:
                    out_links[doc_id].add(target)
                    in_links[target].add(doc_id)

        # Power-iteration PageRank
        scores = {doc_id: 1.0 / N for doc_id in range(N)}
        d = PAGERANK_DAMPING

        for _ in range(PAGERANK_ITERATIONS):
            new_scores = {}
            for doc_id in range(N):
                rank = (1.0 - d) / N
                for src in in_links[doc_id]:
                    out_count = len(out_links[src])
                    if out_count > 0:
                        rank += d * scores[src] / out_count
                new_scores[doc_id] = rank
            scores = new_scores

        # Normalize to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {doc_id: s / max_score for doc_id, s in scores.items()}

        with open(PAGERANK_FILE, "wb") as f:
            pickle.dump(scores, f)
        print(f"Saved PageRank scores for {N} documents.")
        return scores

    def compute_analytics(self, term_dict):
        total_size = (
            os.path.getsize(POSTINGS_FILE) +
            os.path.getsize(TERM_DICT_FILE) +
            os.path.getsize(MAPPING_FILE)
        )
        print("\n===== INDEX ANALYTICS =====")
        print(f"Documents indexed:          {self.doc_count}")
        print(f"Unique tokens:              {len(term_dict)}")
        print(f"Index size (KB):            {round(total_size / 1024, 2)}")
        print(f"Exact duplicates skipped:   {self.duplicate_count}")
        print(f"Near-duplicates skipped:    {self.near_duplicate_count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python indexer.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    indexer = Indexer()
    indexer.process_directory(dataset_path)
    term_dict = indexer.merge_partials()
    indexer.compute_pagerank()
    indexer.compute_analytics(term_dict)
