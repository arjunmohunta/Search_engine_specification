import pickle
from indexer import tokenize
from constants import MAPPING_FILE, POSTINGS_FILE, TERM_DICT_FILE

def get_query():
    query = input("What would you like to search for?\n")
    query_tokens = tokenize(query)
    return query_tokens

def get_postings(terms: list) -> dict:
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

    return postings

def get_term_info(terms: list) -> dict:
    info = {}
    with open(TERM_DICT_FILE, "rb") as f:
        term_dict = pickle.load(f)
        for term in terms:
            offset = term_dict.get(term, None)
            info[term] = offset
    return info

def get_urls(docs: list[tuple]) -> list:
    mapping = None
    with open(MAPPING_FILE, "rb") as f:
        mapping = pickle.load(f)
    
    urls = [mapping[posting[0]] for posting in docs[:5]]
    return urls

def search(query_tokens):
    term_postings = get_postings(query_tokens)

    # processing queries as AND queries
    # returns a list of tuples (term, posting) -- posting in format {docID: tf_score}
    sorted_terms = sorted(term_postings.items(), key=lambda item: len(item[1])) 
    common_docs = list(sorted_terms[0][1].keys())
    docs_tf = {}

    for term in sorted_terms:
        term_docs = term[1].keys()
        common_docs = list(set(common_docs) & set(term_docs))

    for term in sorted_terms:
        posting = term[1]
        for docID in common_docs:
            if docID not in docs_tf.keys():
                docs_tf[docID] = 0
            docs_tf[docID] += posting[docID]["tf"]
    
    sorted_docs = sorted(docs_tf.items(), key=lambda item: item[1], reverse = True)
    
    top_5 = get_urls(sorted_docs)
    return top_5

if __name__ == "__main__":
    query_tokens = get_query()
    results = search(query_tokens)

    print("Here are the top 5 results")
    for i, url in enumerate(results):
        print(f"{i+1}: {url}")