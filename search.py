import math
import pickle
import os
from indexer import tokenize
from constants import MAPPING_FILE, POSTINGS_FILE, TERM_DICT_FILE

term_dict = None

def file_check():
    if not os.path.exists(MAPPING_FILE) or not os.path.exists(TERM_DICT_FILE) or not os.path.exists(POSTINGS_FILE):
        print("Index files not found. Run indexer first.")
        exit(1)
    global term_dict
    with open(TERM_DICT_FILE, "rb") as f:
        term_dict = pickle.load(f)

def get_query():
    query = input("What would you like to search for?\n")
    query_tokens = tokenize(query)
    return query_tokens

def get_postings(terms: list) -> tuple[dict, dict]:
    """Returns (postings_by_term, term_info_by_term). term_info is (offset, length) or (offset, length, df)."""
    info = get_term_info(terms)
    postings = {}

    with open(POSTINGS_FILE, "rb") as f:
        for term in terms:
            term_info = info[term]
            if term_info is None:
                postings[term] = {}
            else:
                offset = term_info[0]
                length = term_info[1]
                f.seek(offset)
                data = f.read(length)
                postings[term] = pickle.loads(data)
    return postings, info

def get_term_info(terms: list) -> dict:
    """Returns for each term: (offset, length, df) or None. Supports old format (offset, length)."""
    global term_dict
    info = {}
    if term_dict is not None:
        for term in terms:
            val = term_dict.get(term, None)
            info[term] = val
    return info

def load_mapping_and_n():
    """Load (mapping, doc_count N) for URL lookup and TF-IDF."""
    with open(MAPPING_FILE, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        mapping, doc_count = data
    else:
        mapping = data
        doc_count = len(mapping)
    return mapping, doc_count

def get_urls(docs: list[tuple], mapping: dict) -> list:
    return [mapping[doc_id] for doc_id, _ in docs[:5]]

def tf_idf_score(N: int, wt: float, df: int) -> float:
    """TF-IDF component: (1 + log(1 + wt)) * log((N+1)/(df+1)). Uses weighted tf for important words."""
    if df <= 0:
        return 0.0
    tf_comp = 1.0 + math.log(1.0 + wt)
    idf = math.log((N + 1) / (df + 1))
    return tf_comp * idf


def search(query_tokens, mapping: dict, N: int):
    """
    AND query + TF-IDF ranking with important-words weighting (wt stored by indexer).
    Returns top 5 URLs.
    """
    term_postings, term_info = get_postings(query_tokens)
    # AND: start with smallest posting list
    sorted_terms = sorted(term_postings.items(), key=lambda item: len(item[1]))
    if not sorted_terms or len(sorted_terms[0][1]) == 0:
        return []
    common_docs = set(sorted_terms[0][1].keys())
    for term, posting in sorted_terms[1:]:
        common_docs &= set(posting.keys())
    if not common_docs:
        return []

    # Score each doc by sum of TF-IDF over query terms (using weighted tf)
    doc_scores = {}
    for term, posting in sorted_terms:
        val = term_info.get(term)
        if val is None:
            df = 1
        else:
            df = val[2] if len(val) >= 3 else len(posting) if posting else 1
        for doc_id in common_docs:
            if doc_id not in posting:
                continue
            entry = posting[doc_id]
            wt = entry.get("wt", entry.get("tf", 0))
            score = tf_idf_score(N, wt, df)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return get_urls(sorted_docs, mapping)

def main():
    file_check()
    mapping, N = load_mapping_and_n()
    while True:
        query_tokens = get_query()
        if "exit" in query_tokens:
            break
        if not query_tokens:
            print("Please enter a valid query.\n")
            continue

        results = search(query_tokens, mapping, N)
        if not results:
            print("No results found.")
        else:
            print("Here are the top 5 results")
            for i, url in enumerate(results):
                print(f"{i+1}: {url}")

if __name__ == "__main__":
    main()