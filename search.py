import math
import pickle
import os
from indexer import tokenize
from constants import MAPPING_FILE, POSTINGS_FILE, TERM_DICT_FILE

term_dict = None
TOP_K_RESULTS = 10

def check_index_files():
    global term_dict
    if not all(os.path.exists(f) for f in [MAPPING_FILE, POSTINGS_FILE, TERM_DICT_FILE]):
        print("Index files not found. Run indexer first.")
        exit(1)
    with open(TERM_DICT_FILE, "rb") as f:
        term_dict = pickle.load(f)

def load_mapping_and_doc_count():
    with open(MAPPING_FILE, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        mapping, doc_count = data
    else:
        mapping = data
        doc_count = len(mapping)
    return mapping, doc_count

def get_query_tokens():
    query = input("Enter search query (or 'exit' to quit):\n")
    return tokenize(query)

def get_term_info(terms):
    info = {}
    for term in terms:
        val = term_dict.get(term)
        if val:
            if len(val) == 3:
                info[term] = {"offset": val[0], "length": val[1], "df": val[2]}
            else:
                info[term] = {"offset": val[0], "length": val[1], "df": None}
        else:
            info[term] = None
    return info

def get_postings(terms):
    postings = {}
    term_info = get_term_info(terms)
    with open(POSTINGS_FILE, "rb") as f:
        for term in terms:
            info = term_info.get(term)
            if info is None:
                postings[term] = {}
            else:
                f.seek(info["offset"])
                data = f.read(info["length"])
                postings[term] = pickle.loads(data)
    return postings, term_info

def tf_idf_score(N, wt, df):
    if df <= 0: return 0.0
    return (1.0 + math.log(1.0 + wt)) * math.log((N+1)/(df+1))

def rank_documents(query_tokens, postings, term_info, N):
    doc_scores = {}
    for term in query_tokens:
        post = postings.get(term, {})
        info = term_info.get(term)
        df = info["df"] if info and info.get("df") is not None else len(post) if post else 1
        for doc_id, values in post.items():
            wt = values.get("wt", values.get("tf",0))
            doc_scores[doc_id] = doc_scores.get(doc_id,0) + tf_idf_score(N, wt, df)
    sorted_docs = sorted(doc_scores.items(), key=lambda x:x[1], reverse=True)
    return sorted_docs

def get_top_urls(sorted_docs, mapping, top_k=TOP_K_RESULTS):
    return [mapping[doc_id] for doc_id,_ in sorted_docs[:top_k]]

def search_engine():
    check_index_files()
    mapping, N = load_mapping_and_doc_count()
    print(f"Loaded index with {N} documents.")

    while True:
        query_tokens = get_query_tokens()
        if "exit" in query_tokens:
            print("Exiting search engine.")
            break
        if not query_tokens:
            print("Please enter a valid query.\n")
            continue
        postings, term_info = get_postings(query_tokens)
        sorted_docs = rank_documents(query_tokens, postings, term_info, N)
        if not sorted_docs:
            print("No results found.")
        else:
            urls = get_top_urls(sorted_docs, mapping)
            print(f"Top {len(urls)} results:")
            for i,url in enumerate(urls):
                print(f"{i+1}: {url}")
        print("-"*40)

if __name__ == "__main__":
    search_engine()
