"""Compare retrieval on the eight questions, without the cost of a full RAGAS grid."""

import time

import numpy as np
import pandas as pd

from taller_3.core import OUT, errors, save
from taller_3.experiment import load_inputs
from taller_3.retrieval import Collection, encoder


def main():
    with errors("embedding_trial"):
        config, questions = load_inputs()
        directory = OUT / "embedding-trial"
        save(directory / "manifest.json", {
            "scope": "retrieval only, not RAGAS or final system comparison",
            "config": config, "questions": questions,
            "fixed_tokenizer": config["tokenizers"][0], "fixed_chunker": "character",
        })
        rows = []
        for name in config["embeddings"]:
            start = time.perf_counter()
            collection = Collection(config, name, config["tokenizers"][0], "character")
            elapsed = time.perf_counter() - start
            assert len({doc["source"] for doc in collection.documents}) == 10
            assert np.isfinite(collection.matrix).all()
            np.testing.assert_allclose(np.linalg.norm(collection.matrix, axis=1), 1, atol=1e-5)
            save(directory / f"{name.replace('/', '--')}.json", {
                "embedding": name, "revision": config["hf_revisions"][name],
                "dimensions": collection.matrix.shape[1], "chunks": collection.documents,
                "load_and_index_seconds": elapsed, "prompts": encoder(name).prompts,
            })
            for question in questions:
                for k in config["top_k"]:
                    start = time.perf_counter()
                    docs = collection.search(question["question"], k)
                    seconds = time.perf_counter() - start
                    retrieved = {doc["source"] for doc in docs}
                    expected = set(question["sources"])
                    rows.append({"embedding": name, "id": question["id"], "type": question["type"],
                                 "top_k": k, "source_recall": len(retrieved & expected) / len(expected),
                                 "source_hit": int(bool(retrieved & expected)), "seconds": seconds,
                                 "retrieved_sources": sorted(retrieved), "expected_sources": sorted(expected)})
            pd.DataFrame(rows).to_csv(directory / "per_question.csv", index=False)
            print(f"{name}: {collection.matrix.shape}, all ten files indexed", flush=True)
        summary = pd.DataFrame(rows).groupby(["embedding", "top_k"])[
            ["source_recall", "source_hit", "seconds"]].mean()
        summary.to_csv(directory / "summary.csv")
        print(summary.round(4).to_string(), flush=True)
        from taller_3.plots import generate
        generate()


if __name__ == "__main__":
    main()
