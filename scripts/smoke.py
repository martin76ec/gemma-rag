"""Real integration check, explicitly separate from the eight class questions."""

import asyncio
import itertools

from taller_3.core import OUT, ROOT, errors, read, save
from taller_3.experiment import Judge
from taller_3.retrieval import Collection, tokenizer
from taller_3.systems import SYSTEMS, run_system

with errors("smoke"):
    config = read(ROOT / "config.json")
    question = {
        "id": "smoke",
        "type": "factual",
        "question": "¿Cuál es la autonomía del LR-200?",
        "reference": "El Lumina LR-200 tiene una autonomía aproximada de 180 minutos.",
    }
    inventories = []
    for emb, tok, kind in itertools.product(
        config["embeddings"], config["tokenizers"], config["chunkers"]
    ):
        collection = Collection(config, emb, tok, kind)
        assert len({d["source"] for d in collection.documents}) == 10
        inventories.append(
            {
                "embedding": emb,
                "tokenizer": tok,
                "chunker": kind,
                "vocab_size": len(tokenizer(tok)),
                "chunks": len(collection.documents),
            }
        )
        print(inventories[-1], flush=True)
    save(OUT / "smoke" / "collections.json", inventories)
    judge = Judge(config)
    for system in SYSTEMS:
        result = run_system(system, question["question"], collection, config, 2)
        save(
            OUT / "smoke" / f"{SYSTEMS.index(system)}.json",
            {"not_class_results": True, **question, **result},
        )
        result.update(asyncio.run(judge.score(question, result)))
        save(
            OUT / "smoke" / f"{SYSTEMS.index(system)}.json",
            {"not_class_results": True, **question, **result},
        )
        print(system, result["answer_correctness"], result["faithfulness"], flush=True)
