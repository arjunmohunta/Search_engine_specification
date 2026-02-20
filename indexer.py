import os
import json
import pickle
import re
import sys
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

PARTIAL_DUMP_THRESHOLD = 10000
INDEX_DIR = "index_files"
MAPPING_FILE = "url_mappings.json"
os.makedirs(INDEX_DIR, exist_ok=True)

stemmer = PorterStemmer()


def tokenize(text):
    # do we need to worry about apostrophes? maybe just strip()
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [stemmer.stem(token) for token in tokens]


def extract_text_and_importance(html):
    try:
        soup = BeautifulSoup(html, "html.parser")

        important_text = ""

        if soup.title:
            important_text += soup.title.get_text() + " "

        for tag in soup.find_all(["h1", "h2", "h3"]):
            important_text += tag.get_text() + " "

        for tag in soup.find_all(["b", "strong"]):
            important_text += tag.get_text() + " "

        normal_text = soup.get_text()

        return normal_text, important_text

    except Exception:
        # If parsing fails, safely return empty strings
        return "", ""


class Indexer:
    def __init__(self):
        self.index = {}
        self.mapping = {}
        self.doc_count = 0
        self.partial_index_count = 0

    def add_document(self, doc_id, content):
        normal_text, important_text = extract_text_and_importance(content)

        normal_tokens = tokenize(normal_text)
        important_tokens = tokenize(important_text)

        for token in normal_tokens:
            if token not in self.index:
                self.index[token] = {}
            if doc_id not in self.index[token]:
                self.index[token][doc_id] = {"tf": 0, "importance": 0}
            self.index[token][doc_id]["tf"] += 1

        for token in important_tokens:
            if token not in self.index:
                self.index[token] = {}
            if doc_id not in self.index[token]:
                self.index[token][doc_id] = {"tf": 0, "importance": 0}
            self.index[token][doc_id]["importance"] += 1

    def flush_partial_index(self):
        filename = os.path.join(
            INDEX_DIR, f"partial_{self.partial_index_count}.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(self.index, f)

        print(f"Flushed partial index {self.partial_index_count}")

        self.partial_index_count += 1
        self.index = {}  # Reset safely

    def process_directory(self, root_dir):
        for root, dirs, files in os.walk(root_dir):
            # iterate through the directories
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                for dir, subdirs, dir_files in os.walk(dir_path):
                    # getting json files within the directory
                    for file in dir_files:
                        if file.endswith(".json"):
                            path = os.path.join(dir, file)

                            try:
                                with open(path, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                            except Exception:
                                continue  # Skip bad JSON files

                            content = data.get("content", "")
                            url = data.get("url", "")

                            if content and url: # add url, content if both url and content are not empty
                                doc_id = self.doc_count
                                self.doc_count += 1

                                self.mapping[url] = doc_id

                                self.add_document(doc_id, content)

                                if len(self.index) >= PARTIAL_DUMP_THRESHOLD:
                                    self.flush_partial_index()

        # Final flush
        if self.index:
            self.flush_partial_index()
        
        # store url mapping
        with open(MAPPING_FILE, "w") as f:
            json.dump(self.mapping)

    def merge_partials(self):
        merged_index = {}

        for i in range(self.partial_index_count):
            filename = os.path.join(INDEX_DIR, f"partial_{i}.pkl")

            with open(filename, "rb") as f:
                partial = pickle.load(f)

            for term, postings in partial.items():
                if term not in merged_index:
                    merged_index[term] = {}

                for doc_id, values in postings.items():
                    if doc_id not in merged_index[term]:
                        merged_index[term][doc_id] = {
                            "tf": 0,
                            "importance": 0,
                        }

                    merged_index[term][doc_id]["tf"] += values["tf"]
                    merged_index[term][doc_id]["importance"] += values["importance"]

        final_path = os.path.join(INDEX_DIR, "final_index.pkl")
        with open(final_path, "wb") as f:
            pickle.dump(merged_index, f)

        print("Final index saved.")
        return merged_index


def compute_analytics(indexer, merged_index):
    num_docs = indexer.doc_count
    num_tokens = len(merged_index)

    total_size = 0
    for file in os.listdir(INDEX_DIR):
        total_size += os.path.getsize(os.path.join(INDEX_DIR, file))

    total_size_kb = total_size / 1024

    print("\n===== INDEX ANALYTICS =====")
    print("Number of indexed documents:", num_docs)
    print("Number of unique tokens:", num_tokens)
    print("Total index size (KB):", round(total_size_kb, 2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python indexer.py <path_to_dataset>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    indexer = Indexer()
    indexer.process_directory(dataset_path)
    merged_index = indexer.merge_partials()
    compute_analytics(indexer, merged_index)
