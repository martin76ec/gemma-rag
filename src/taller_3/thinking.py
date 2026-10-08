"""Paired thinking ablation with fixed retrieval, judge and generation budget."""

import hashlib
import json
from importlib.metadata import version

import pandas as pd

from .core import OUT, ROOT, errors, save
from .experiment import Judge, evaluate_one, load_inputs
from .plots import plot_results
from .retrieval import Collection
from .systems import LLM


def compare(systems=None, embedding="lightonai/mDenseOn", top_k=4):
    with errors("thinking_comparison"):
        config, questions = load_inputs()
        systems = systems or ["RAG simple"]
        selection = {"embedding": embedding, "tokenizer": config["tokenizers"][0],
                     "chunker": "character", "top_k": top_k}
        manifest = {
            "kind": "thinking_comparison", "config": config, "selection": selection,
            "systems": systems, "questions": questions,
            "judge_think": False, "order": "off/on for even questions, on/off for odd questions",
            "selection_basis": "fixed candidate from retrieval trial, not RAGAS grid winner",
            "model": LLM(config).client.show(config["model"]).model_dump(mode="json"),
            "packages": {p: version(p) for p in ("ragas", "ollama", "sentence-transformers", "transformers")},
            "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for pattern in ("data/*.txt", "src/taller_3/*.py") for p in sorted(ROOT.glob(pattern))},
        }
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
        directory = OUT / f"thinking-{digest}"
        save(directory / "manifest.json", manifest)
        save(OUT / "latest-thinking.json", {"run": directory.name, "status": "running"})
        collection = Collection(config, embedding, selection["tokenizer"], "character")
        save(directory / "collection.json", collection.documents)
        judge = Judge(config)
        # Model load/initialization is excluded from the timed paired observations.
        for mode in (False, True):
            LLM({**config, "think": mode}).ask("Responde solo: listo.")
        records = []
        for system_index, system in enumerate(systems):
            for index, question in enumerate(questions):
                for mode in ((False, True) if index % 2 == 0 else (True, False)):
                    print(f"{system} · {question['id']} · think={mode}", flush=True)
                    result = evaluate_one(directory, f"s{system_index}-think-{int(mode)}", system,
                                          question, collection, {**config, "think": mode}, top_k, judge)
                    records.append({**result, "think": mode,
                                    "variant": f"{system} · {'con' if mode else 'sin'} thinking"})
                    pd.DataFrame(records).drop(columns=["calls", "trace", "documents", "retrieved_contexts"]).to_csv(
                        directory / "per_question.csv", index=False)
                    print(f"  correctness={result['answer_correctness']:.3f}, "
                          f"faithfulness={result['faithfulness']:.3f}, "
                          f"tokens={result['input_tokens'] + result['output_tokens']}", flush=True)
        frame = pd.DataFrame(records)
        frame["total_tokens"] = frame.input_tokens + frame.output_tokens
        metrics = ["answer_correctness", "faithfulness", "total_tokens", "llm_calls", "seconds"]
        off = frame[~frame.think].set_index(["system", "id"])
        on = frame[frame.think].set_index(["system", "id"])
        if "RAG simple" in systems:
            for question in questions:
                pair = ("RAG simple", question["id"])
                assert off.loc[pair, "documents"] == on.loc[pair, "documents"], "Retrieval differs between modes"
        delta = on[metrics] - off[metrics]
        delta.to_csv(directory / "paired_deltas.csv")
        summary = frame.groupby(["system", "think"])[metrics].mean()
        summary.to_csv(directory / "summary.csv")
        save(directory / "results.json", records)
        plot_results(frame, directory, "Thinking on/off · 8 preguntas pareadas", group="variant")
        report = f"""# Comparación de pensamiento nativo de Gemma

Colección fija: `{selection}`. Generador `{config['model']}`, temperatura {config['temperature']},
semilla {config['seed']}, num_predict {config['num_predict']} en ambos modos. Juez sin thinking.
Orden alternado por pregunta; calentamiento excluido. Referencias visibles solo al juez.
Configuración candidata del ensayo de recuperación, no ganadora de la grilla RAGAS.

## Medias por pregunta

{summary.round(3).to_markdown()}

## Diferencia pareada media: activado menos desactivado

{delta.groupby('system').mean().round(3).to_markdown()}

Una diferencia positiva en correctness/faithfulness indica mejor puntuación;
una diferencia positiva en tokens, llamadas o segundos indica mayor costo.
Para RAG simple se verificó igualdad exacta de documentos recuperados entre modos.
En sistemas con llamadas intermedias, activar thinking también puede cambiar el enrutamiento o las búsquedas.
Ocho preguntas representativas, una observación por modo y pregunta, juez LLM del mismo modelo:
resultados descriptivos, sin prueba de significancia ni estimación de generalización.
Las cachés pueden influir en latencia; el orden alternado no elimina esa variación.

![Calidad](quality.png)

![Costos](costs.png)

![Calidad frente a tokens](correctness_vs_tokens.png)
"""
        (directory / "README.md").write_text(report)
        save(OUT / "latest-thinking.json", {"run": directory.name, "status": "complete"})
        print(summary.round(3).to_string(), flush=True)
        print(f"Informe: {directory / 'README.md'}", flush=True)
        return directory
