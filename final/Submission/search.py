import math
import pickle
import os
from indexer import tokenize
from constants import MAPPING_FILE, POSTINGS_FILE, TERM_DICT_FILE, PAGERANK_FILE, PAGERANK_ALPHA, HITS_FILE, HITS_ALPHA, TOP_K_RESULTS

term_dict = None
pagerank_scores = None
hits_scores = None

def check_index_files():
    # checks that files needed for search engine exist
    global term_dict, pagerank_scores, hits_scores
    if not all(os.path.exists(f) for f in [MAPPING_FILE, POSTINGS_FILE, TERM_DICT_FILE]):
        print("Index files not found. Run indexer first.")
        exit(1)
    with open(TERM_DICT_FILE, "rb") as f:
        term_dict = pickle.load(f)
    if os.path.exists(PAGERANK_FILE):
        with open(PAGERANK_FILE, "rb") as f:
            pagerank_scores = pickle.load(f)
    if os.path.exists(HITS_FILE):
        with open(HITS_FILE, "rb") as f:
            hits_scores = pickle.load(f)

def load_mapping_and_doc_count():
    # loads document->id mapping once
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

def get_bigrams(tokens):
    """Generate bigram strings from token list: 'word1_word2'."""
    return [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]

def get_term_info(terms):
    """
    Grab the offset, length, and doc freq. of the terms
    """
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
    """
    Grab the postings of each term based on the offset and length of term data
    """
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
    """
    Calculates the tf-idf score
    """
    if df <= 0:
        return 0.0
    return (1.0 + math.log(1.0 + wt)) * math.log((N + 1) / (df + 1))

def is_phrase_match(postings, query_tokens, doc_id):
    """Return True if query_tokens appear at consecutive body positions in doc_id.

    Note: positions are only recorded for body tokens in the index. Terms that appear
    exclusively in titles/headings/bold text do not carry positional information, so
    phrase matching is intentionally limited to the document body.
    """
    pos_sets = []
    for term in query_tokens:
        entry = postings.get(term, {}).get(doc_id)
        if not entry or not entry.get("pos"):
            return False
        pos_sets.append(set(entry["pos"]))
    # Look for anchor position p such that p+i in pos_sets[i] for all i
    for p in pos_sets[0]:
        if all(p + i in pos_sets[i] for i in range(1, len(query_tokens))):
            return True
    return False

def rank_documents(query_tokens, postings, term_info, N, mapping, query_bigrams=None):
    doc_scores = {}
    for term in query_tokens:
        # get term posting and document frequency
        post = postings.get(term, {})
        info = term_info.get(term)
        df = info["df"] if info and info.get("df") is not None else (len(post) if post else 1)
        for doc_id, values in post.items():
            # update document scores to include tf-idf scores
            wt = values.get("wt", values.get("tf", 0))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + tf_idf_score(N, wt, df)

    # Bigram scores with 1.5x weight multiplier
    if query_bigrams:
        for term in query_bigrams:
            # get bigram posting and document frequency
            post = postings.get(term, {})
            info = term_info.get(term)
            df = info["df"] if info and info.get("df") is not None else (len(post) if post else 1)
            for doc_id, values in post.items():
                # update document score to include tf-idf score with 1.5 weight for bigrams
                wt = values.get("wt", values.get("tf", 0))
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.5 * tf_idf_score(N, wt, df)

    # Penalize raw file URLs so HTML pages tend to outrank dataset files.
    # Penalties stack (multiply together).
    WEB_EXTENSIONS = (".html", ".htm", ".php")
    DATASET_TXT_KEYWORDS = ("random", "dvd", "maven", "contents")

    for doc_id in doc_scores:
        url = mapping.get(doc_id, "")
        mult = 1.0
        if url.endswith(".txt") or url.endswith(".bib") or url.endswith(".ff"):
            mult *= 0.1
        if "/datasets/" in url:
            mult *= 0.2
        if url.endswith(".txt"):
            url_lower = url.lower()
            if any(kw in url_lower for kw in DATASET_TXT_KEYWORDS):
                mult *= 0.15
        if not url.endswith(WEB_EXTENSIONS):
            mult *= 0.4
        doc_scores[doc_id] *= mult

    # Blend in PageRank
    if pagerank_scores:
        for doc_id in doc_scores:
            # update document score to include pagerank weight
            pr = pagerank_scores.get(doc_id, 0.0)
            doc_scores[doc_id] *= (1.0 + PAGERANK_ALPHA * pr)

    # Blend in HITS authority
    if hits_scores:
        for doc_id in doc_scores:
            # update document score to include HITS weight
            authority = hits_scores.get(doc_id, {}).get("authority", 0.0)
            doc_scores[doc_id] *= (1.0 + HITS_ALPHA * authority)

    # Phrase bonus: 2× score if all tokens appear consecutively (positional index)
    if len(query_tokens) > 1:
        for doc_id in list(doc_scores.keys()):
            # Upweight the doc score if query terms appear in sequence
            if is_phrase_match(postings, query_tokens, doc_id):
                doc_scores[doc_id] *= 2.0

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True) # sort scores in descending order
    return sorted_docs

def get_top_urls(sorted_docs, mapping, top_k=TOP_K_RESULTS):
    """
    Return the top k urls
    """
    seen = set()
    urls = []
    for doc_id, _ in sorted_docs:
        url = mapping[doc_id]
        if url not in seen:
            seen.add(url)
            urls.append(url)
        if len(urls) >= top_k:
            break
    return urls

def search_engine():
    check_index_files()
    mapping, N = load_mapping_and_doc_count()
    pr_status  = "with PageRank" if pagerank_scores else "without PageRank"
    print(f"Loaded index with {N} documents ({pr_status}).")

    while True:
        query_tokens = get_query_tokens()
        # query validity check
        if "exit" in query_tokens:
            print("Exiting search engine.")
            break
        if not query_tokens:
            print("Please enter a valid query.\n")
            continue
        
        # combine single-term tokens and bigram tokens
        query_bigrams = get_bigrams(query_tokens)
        all_terms = query_tokens + query_bigrams
        # get all postings of all tokens
        postings, term_info = get_postings(all_terms)

        # rank documents and output results
        sorted_docs = rank_documents(query_tokens, postings, term_info, N, mapping, query_bigrams)
        if not sorted_docs:
            print("No results found.")
        else:
            urls = get_top_urls(sorted_docs, mapping)
            print(f"Top {len(urls)} results:")
            for i, url in enumerate(urls):
                print(f"{i+1}: {url}")
        print("-" * 40)

if __name__ == "__main__":
    search_engine()
