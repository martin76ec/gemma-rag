"""Resumable grid and comparison; judge calls are outside measured system costs."""

import asyncio
import hashlib
import itertools
import json
import math
from importlib.metadata import version

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerCorrectness, Faithfulness
from ragas.run_config import RunConfig

from .core import OUT, ROOT, LabError, errors, read, save
from .retrieval import Collection, vectors
from .systems import LLM, SYSTEMS, run_system


class JudgeEmbeddings(Embeddings):
    def __init__(self, name):
        self.name = name

    def embed_documents(self, texts):
        return vectors(self.name, texts).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class JudgeChat(ChatOllama):
    def _chat_params(self, messages, stop=None, **kwargs):
        return {**super()._chat_params(messages, stop=stop, **kwargs), "think": False}


class Judge:
    def __init__(self, config):
        llm = LangchainLLMWrapper(JudgeChat(
            model=config["judge_model"], base_url=config["ollama_url"],
            temperature=0, seed=config["seed"], num_ctx=8192, num_predict=2048,
            keep_alive="30m"), run_config=RunConfig(timeout=600, max_retries=2))
        embedding = LangchainEmbeddingsWrapper(JudgeEmbeddings(config["embeddings"][0]))
        self.metrics = [AnswerCorrectness(llm=llm, embeddings=embedding), Faithfulness(llm=llm)]
        for metric in self.metrics:
            metric.init(RunConfig(timeout=600, max_retries=2))

    async def score(self, question, result):
        sample = SingleTurnSample(user_input=question["question"], reference=question["reference"],
                                  response=result["response"], retrieved_contexts=result["retrieved_contexts"])
        scores = {}
        for metric in self.metrics:
            # The judge emits JSON as free text; it occasionally breaks escaping
            # (e.g. an unescaped quote from the source text) and fails parsing,
            # not scoring. Retry that formatting failure instead of losing the run.
            for attempt in itertools.count(1):
                try:
                    score = float(await metric.single_turn_ascore(sample))
                    break
                except Exception:
                    if attempt >= 5:
                        raise
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise LabError(f"Métrica {metric.name} inválida: {score}")
            scores[metric.name] = score
        return scores


def load_inputs():
    config = read(ROOT / "config.json")
    path = ROOT / config["questions_path"]
    if not path.exists():
        raise LabError("Falta questions.json con las ocho preguntas de clase, referencias y tipos")
    questions = read(path)
    if len(questions) != 8 or len({q["id"] for q in questions}) != 8:
        raise LabError("Se requieren exactamente ocho preguntas con IDs únicos")
    for q in questions:
        if any(not q.get(field) for field in ("id", "question", "reference", "type")):
            raise LabError("Cada pregunta necesita id, question, reference y type")
    return config, questions


def evaluate_one(run_dir, key, system, question, collection, config, k, judge):
    path = run_dir / "records" / f"{key}__{question['id']}.json"
    if path.exists():
        record = read(path)
    else:
        record = {"key": key, "system": system, **question,
                  **run_system(system, question["question"], collection, config, k)}
        save(path, record)  # Save generation before calling the judge.
    if "answer_correctness" not in record:
        record.update(asyncio.run(judge.score(question, record)))
        save(path, record)
    return record


def run():
    with errors("experiment"):
        config, questions = load_inputs()
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "data").glob("*.txt"))}
        code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT / "src" / "taller_3").glob("*.py"))}
        packages = {p: version(p) for p in ("ragas", "transformers", "sentence-transformers", "ollama")}
        model_info = LLM(config).client.show(config["model"]).model_dump(mode="json")
        signature = {"config": config, "questions": questions, "files": files,
                     "code": code, "packages": packages, "model": model_info}
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
        run_dir = OUT / digest
        save(run_dir / "manifest.json", signature)
        save(OUT / "latest.json", {"run": digest, "status": "running"})
        judge, grid, grid_records = Judge(config), [], []
        for index, (embedding, tok, kind) in enumerate(itertools.product(
                config["embeddings"], config["tokenizers"], config["chunkers"])):
            collection = Collection(config, embedding, tok, kind)
            save(run_dir / "collections" / f"{index}.json", collection.documents)
            for k in config["top_k"]:
                key = f"grid-{index}-k{k}"
                print(f"Evaluando {key}: {embedding}, {tok}, {kind}", flush=True)
                rows = [evaluate_one(run_dir, key, "RAG simple", q, collection, config, k, judge)
                        for q in questions]
                grid_records.extend(rows)
                grid.append({"key": key, "collection": index, "embedding": embedding, "tokenizer": tok,
                             "chunker": kind, "top_k": k,
                             "answer_correctness": sum(r["answer_correctness"] for r in rows) / 8,
                             "faithfulness": sum(r["faithfulness"] for r in rows) / 8})
                pd.DataFrame(grid).to_csv(run_dir / "grid.csv", index=False)
        best = max(grid, key=lambda row: row["answer_correctness"])
        save(run_dir / "best_config.json", best)
        collection = Collection(config, best["embedding"], best["tokenizer"], best["chunker"])
        records = [r for r in grid_records if r["key"] == best["key"]]
        for i, system in enumerate(SYSTEMS[1:], 1):
            print(f"Comparando {system}", flush=True)
            records.extend(evaluate_one(run_dir, f"system-{i}", system, q, collection,
                                        config, best["top_k"], judge) for q in questions)
        save(run_dir / "results.json", records)
        from .report import build_report
        build_report(run_dir, records, best, config)
        save(OUT / "latest.json", {"run": digest, "status": "complete"})
        return run_dir
