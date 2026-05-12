from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
)


def load_rag_components(db_path=DB_PATH):
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)
    milvus_client = MilvusClient(uri=db_path)

    return embed_model, reranker, milvus_client


def retrieve_context(
    user_query,
    embed_model,
    reranker,
    milvus_client,
    collection_name=COLLECTION_NAME,
    top_k=3,
    score_threshold=0.2,
):
    query_vector = embed_model.encode(user_query).tolist()

    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=top_k,
        output_fields=["ocr_text", "caption", "image_path"],
    )

    hits = search_res[0]

    if len(hits) == 0:
        return []

    rerank_pairs = []

    for hit in hits:
        entity = hit["entity"]
        combined_text = f"{entity['caption']}. {entity['ocr_text']}"
        rerank_pairs.append([user_query, combined_text])

    scores = reranker.predict(rerank_pairs)

    for i, hit in enumerate(hits):
        hit["rerank_score"] = float(scores[i])

    sorted_hits = sorted(
        hits,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    final_hits = [
        hit for hit in sorted_hits
        if hit["rerank_score"] >= score_threshold
    ][:top_k]

    return final_hits