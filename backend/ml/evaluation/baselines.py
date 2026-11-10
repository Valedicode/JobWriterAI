"""The CPU baselines (NLP_PROJECT_PLAN.md §8), each written as a prediction file for `val` and test.

- teacher: the teacher's own aggregate. Validity only; its fidelity ceiling is the test-retest.
- tfidf-ridge: TF-IDF pair features -> the teacher's aggregate, fitted on `train`.
- tfidf-logreg: the same features -> the dataset label, fitted on `train`; score = expected label (0-2).
- resume-prior: the resume's mean `train` label (the global mean for a resume not in `train`).
- majority: the most common `train` label.
- embedding-cosine (--embeddings): cosine of the two texts' text-embedding-3-small vectors, the
  teacher's embedding model. Needs OPENAI_API_KEY; costs a few cents.

Nothing here reads the test labels.

Run from backend/:  python -m ml.evaluation.baselines [--embeddings]
"""
import argparse
import os
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

from ml.config import ARTIFACTS, BACKEND_DIR, TEACHER_LABELS
from ml.dataprep.splits import load_split
from ml.evaluation import predictions
from ml.evaluation.metrics import LABELS, ORDINAL
from ml.labeling.label_files import latest, read_jsonl

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MAX_TOKENS = 8000  # the model's limit is 8,191


def pair_features(vectorizer, resumes: list[str], jobs: list[str]):
    """Both texts' TF-IDF vectors, their element-wise product and their cosine, so a linear model
    sees the overlap of the pair rather than each text alone."""
    r, j = vectorizer.transform(resumes), vectorizer.transform(jobs)  # rows are L2-normalised
    product = r.multiply(j).tocsr()
    cosine = sparse.csr_matrix(np.asarray(product.sum(axis=1)))
    return sparse.hstack([r, j, product, cosine]).tocsr()


def resume_prior(train_resumes: list[str], train_labels: list[int]):
    by_resume = defaultdict(list)
    for resume, label in zip(train_resumes, train_labels):
        by_resume[resume].append(label)
    overall = float(np.mean(train_labels))
    means = {resume: float(np.mean(ls)) for resume, ls in by_resume.items()}
    return lambda resume: means.get(resume, overall)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def embed(texts: list[str]) -> dict[str, np.ndarray]:
    """One vector per distinct text, cached in data/ml/baselines/embeddings.sqlite."""
    import tiktoken
    from dotenv import load_dotenv
    from openai import OpenAI

    from ml.labeling.caches import EmbeddingCache, Store

    load_dotenv(BACKEND_DIR / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY (backend/.env or the shell) for the embedding baseline.")
    client, encoding = OpenAI(), tiktoken.get_encoding("cl100k_base")

    def fetch(batch, model):
        out = []
        for k in range(0, len(batch), 64):
            cut = [encoding.decode(encoding.encode(t)[:EMBEDDING_MAX_TOKENS]) for t in batch[k:k + 64]]
            out += [d.embedding for d in client.embeddings.create(model=model, input=cut).data]
        return out

    (ARTIFACTS / "baselines").mkdir(parents=True, exist_ok=True)
    cached = EmbeddingCache(Store(ARTIFACTS / "baselines" / "embeddings.sqlite", "embeddings")).wrap(fetch)
    distinct = list(dict.fromkeys(texts))
    return dict(zip(distinct, map(np.asarray, cached(distinct, model=EMBEDDING_MODEL))))


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge

    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true", help="also the embedding-cosine baseline (API)")
    args = ap.parse_args()

    hf_train = predictions.hf_split("train")
    hf_test = predictions.hf_split("test").select_columns(["resume_text", "job_description_text"])
    train_ids = load_split()["train"]
    teacher_train = latest(read_jsonl(TEACHER_LABELS / "train.jsonl"))
    teacher_test = latest(read_jsonl(TEACHER_LABELS / "test.jsonl"))

    def texts(ds, ids):
        resumes, jobs = ds["resume_text"], ds["job_description_text"]
        return [resumes[i] for i in ids], [jobs[i] for i in ids]

    train_r, train_j = texts(hf_train, train_ids)
    train_labels = [ORDINAL[hf_train["label"][i]] for i in train_ids]
    train_teacher = [teacher_train[i]["report"]["aggregate_score"] for i in train_ids]
    evaluated = {"val": (hf_train, teacher_train), "test": (hf_test, teacher_test)}
    rows = {s: predictions.row_ids(s) for s in predictions.SPLITS}
    pairs = {s: texts(evaluated[s][0], rows[s]) for s in predictions.SPLITS}

    def emit(system, scale, score_of, label_of=None):
        out = []
        for s in predictions.SPLITS:
            scores = score_of(s)
            for k, i in enumerate(rows[s]):
                row = {"row_id": i, "split": s, "score": float(scores[k])}
                if label_of is not None:
                    row["label"] = label_of
                out.append(row)
        predictions.write(system, scale, out)

    emit("teacher", "teacher",
         lambda s: [evaluated[s][1][i]["report"]["aggregate_score"] for i in rows[s]])

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50_000, stop_words="english")
    vectorizer.fit(list(dict.fromkeys(train_r + train_j)))
    x_train = pair_features(vectorizer, train_r, train_j)
    x = {s: pair_features(vectorizer, *pairs[s]) for s in predictions.SPLITS}

    ridge = Ridge(alpha=1.0).fit(x_train, train_teacher)
    emit("tfidf-ridge", "teacher", lambda s: ridge.predict(x[s]))

    logreg = LogisticRegression(max_iter=2000).fit(x_train, train_labels)
    emit("tfidf-logreg", "other", lambda s: logreg.predict_proba(x[s]) @ logreg.classes_)

    prior = resume_prior(train_r, train_labels)
    emit("resume-prior", "other", lambda s: [prior(r) for r in pairs[s][0]])

    majority = Counter(train_labels).most_common(1)[0][0]
    print(f"majority class: {LABELS[majority]}")
    emit("majority", "other", lambda s: [majority] * len(rows[s]), label_of=majority)

    if args.embeddings:
        vectors = embed([t for s in predictions.SPLITS for side in pairs[s] for t in side])
        emit("embedding-cosine", "other", lambda s: cosine_rows(
            np.stack([vectors[t] for t in pairs[s][0]]), np.stack([vectors[t] for t in pairs[s][1]])))


if __name__ == "__main__":
    main()
