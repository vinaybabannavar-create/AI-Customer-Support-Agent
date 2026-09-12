"""
Retrieval/grounding layer: given an incoming customer message, find the
K most similar historically-resolved (customer_message -> brand_reply)
threads. These become the "grounding evidence" the drafting prompt is
required to cite/paraphrase from, instead of freewheeling.

Uses plain TF-IDF + cosine similarity (sklearn) rather than an embedding
API -- see decision_log.md decision #6: this is a small, fast, offline,
zero-cost baseline, and full transparency (max index size). Swapping in
an embedding model later is a one-function change (see NOTES at bottom).
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PROCESSED_THREADS


class ThreadIndex:
    def __init__(self, threads_path=PROCESSED_THREADS):
        self.threads = []
        with open(threads_path, encoding="utf-8") as f:
            for line in f:
                self.threads.append(json.loads(line))
        corpus = [t["customer_message"] for t in self.threads]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def top_k(self, query: str, k: int = 3, exclude_thread_id: str = None):
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: -sims[i])
        out = []
        for i in ranked:
            t = self.threads[i]
            if exclude_thread_id and t["thread_id"] == exclude_thread_id:
                continue
            if sims[i] <= 0:
                break
            out.append({**t, "similarity": float(sims[i])})
            if len(out) >= k:
                break
        return out


# NOTES on swapping to embeddings later:
#   Replace TfidfVectorizer/cosine_similarity with an embedding call
#   (e.g. voyage-3 or an OpenAI/Anthropic embedding endpoint) + a vector
#   index (faiss/chromadb). Interface (top_k) stays identical, so nothing
#   downstream (agent.py) needs to change.
