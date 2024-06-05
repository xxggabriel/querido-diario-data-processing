import os
from typing import Dict, List
import sentence_transformers
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from .interfaces import IndexInterface
from .utils import get_documents_with_ids

def embedding_rerank_excerpts(
    theme: Dict, excerpt_ids: List[str], index: IndexInterface
) -> None:
    user_folder = os.environ["HOME"]
    model = sentence_transformers.SentenceTransformer(
        f"{user_folder}/models/bert-base-portuguese-cased"
    )
    queries = get_natural_language_queries(theme)
    queries_vectors = model.encode(queries, convert_to_tensor=True)

    def process_excerpt(excerpt_id: str):
        documents = list(get_documents_with_ids([excerpt_id], index, theme["index"]))
        if documents:
            excerpt = documents[0]["_source"]
            excerpt_vector = model.encode(excerpt["excerpt"], convert_to_tensor=True)
            excerpt_max_score = sentence_transformers.util.semantic_search(
                excerpt_vector, queries_vectors, top_k=1
            )
            excerpt["excerpt_embedding_score"] = excerpt_max_score[0][0]["score"]
            index.index_document(
                excerpt,
                document_id=excerpt["excerpt_id"],
                index=theme["index"],
                refresh=True,
            )
            return excerpt
        else:
            logging.warning(f"No document found for excerpt ID {excerpt_id}")
            return None

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_excerpt, excerpt_id): excerpt_id for excerpt_id in excerpt_ids}

        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exception caught during processing
            except Exception as e:
                excerpt_id = futures[future]
                logging.warning(f"Error processing excerpt ID {excerpt_id}: {e}")
                logging.exception(e)

def get_natural_language_queries(theme: Dict) -> List[str]:
    return [query["title"] for query in theme["queries"]]
