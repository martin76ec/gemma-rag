"""One interactive question, using explicit defaults without a grid or judge."""

from datetime import UTC, datetime

from .core import OUT, ROOT, LabError, errors, read, save
from .retrieval import Collection
from .systems import run_system


def ask_question(question, system="RAG simple", embedding=None, top_k=None):
    with errors("ask"):
        config = read(ROOT / "config.json")
        embedding = embedding or config["embeddings"][0]
        top_k = config["top_k"][0] if top_k is None else top_k
        if not question.strip() or top_k < 1 or embedding not in config["embeddings"]:
            raise LabError("Indique una pregunta, top-k positivo y un embedding de config.json")
        selection = {"embedding": embedding, "tokenizer": config["tokenizers"][0],
                     "chunker": config["chunkers"][0], "top_k": top_k}
        print(f"{system} · {embedding} · top-k={top_k} · thinking={config.get('think', False)}", flush=True)
        collection = Collection(config, embedding, selection["tokenizer"], selection["chunker"])
        result = run_system(system, question, collection, config, top_k)
        path = OUT / "queries" / f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
        save(path, {"question": question, "system": system, "selection": selection,
                    "config": config, "evaluated_with_ragas": False, **result})
        print(result["response"])
        print(f"\n{result['llm_calls']} llamadas · "
              f"{result['input_tokens'] + result['output_tokens']} tokens · {result['seconds']:.2f} s")
        print(f"Respuesta, fuentes y traza: {path.relative_to(ROOT)}")
        return path
