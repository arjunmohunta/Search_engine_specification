"""ICS Search Engine — Streamlit web interface."""
from time import time

import streamlit as st  # type: ignore[import]

from indexer import tokenize
from search import (
    check_index_files,
    load_mapping_and_doc_count,
    get_postings,
    rank_documents,
    get_top_urls,
)


@st.cache_resource
def load_index():
    """Load index files once and reuse across runs."""
    check_index_files()
    return load_mapping_and_doc_count()


def main():
    st.set_page_config(page_title="ICS Search Engine", page_icon="🔍", layout="centered")
    st.title("ICS Search Engine")

    mapping, doc_count = load_index()
    st.caption(f"Indexed documents: {doc_count}")

    query = st.text_input("Search", placeholder="Enter your query", label_visibility="collapsed")
    search_clicked = st.button("Search")

    if search_clicked or (query and query.strip()):
        q = (query or "").strip()
        if not q:
            st.info("Enter a query and click Search.")
            return

        start = time()
        query_tokens = tokenize(q)
        postings, term_info = get_postings(query_tokens)
        sorted_docs = rank_documents(query_tokens, postings, term_info, doc_count, mapping)
        urls = get_top_urls(sorted_docs, mapping)
        elapsed_ms = (time() - start) * 1000.0

        st.metric("Query time", f"{elapsed_ms:.1f} ms")

        if not urls:
            st.write("No results found.")
            return

        st.write("**Top results:**")
        for i, url in enumerate(urls, 1):
            st.markdown(f"{i}. [{url}]({url})")


if __name__ == "__main__":
    import sys
    if "streamlit" in sys.modules:
        main()
    else:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", __file__, "--server.headless", "true", *sys.argv[1:]],
            check=True,
        )
