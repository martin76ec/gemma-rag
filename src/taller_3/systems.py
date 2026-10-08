"""Six strategies share one generator, collection and measured LLM client."""

import json
import re
import time

from ollama import Client

from .core import LabError

SYSTEMS = ("RAG simple", "HyDE", "MRKL", "Self-Ask", "IRCoT", "MRKL con IRCoT")


class LLM:
    def __init__(self, config):
        self.config = config
        self.client = Client(host=config["ollama_url"], timeout=600)
        self.calls = []

    def ask(self, prompt, schema=None):
        result = self.client.chat(
            model=self.config["model"], messages=[{"role": "user", "content": prompt}],
            format=schema, think=self.config.get("think", False), keep_alive="30m",
            options={"temperature": self.config["temperature"], "seed": self.config["seed"],
                     "num_predict": self.config.get("num_predict", 768), "num_ctx": 8192},
        )
        text = result.message.content
        if result.prompt_eval_count is None or result.eval_count is None:
            raise LabError("Ollama no devolvió los conteos de tokens requeridos")
        self.calls.append({"prompt": prompt, "response": text,
                           "thinking_enabled": self.config.get("think", False),
                           "thinking_returned": bool(result.message.thinking),
                           "input_tokens": result.prompt_eval_count,
                           "output_tokens": result.eval_count})
        if result.done_reason == "length":
            raise LabError("La generación alcanzó num_predict; aumente el límite antes de evaluar")
        return json.loads(text) if schema else text


def context(docs):
    return "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)


def responder(llm, question, docs):
    return llm.ask("Responde en español, brevemente y solo con hechos respaldados por los documentos. "
                   "Cita los identificadores entre corchetes. Si falta evidencia, indícalo. "
                   "Si se pregunta por políticas, distingue las históricas de la vigente. "
                   "No añadas comentarios sobre temas ajenos a la pregunta.\n"
                   f"Pregunta: {question}\nDocumentos:\n{context(docs)}")


def enrutador(llm, question, collection):
    sources = sorted({d["source"] for d in collection.documents})
    schema = {"type": "object", "properties": {"sources": {"type": "array", "minItems": 1,
              "items": {"type": "string", "enum": sources}}}, "required": ["sources"]}
    result = llm.ask("Selecciona los archivos necesarios para contestar la pregunta. "
                     "Devuelve sus nombres exactos. Incluye todos los necesarios para razonamiento multihop.\n"
                     f"Pregunta: {question}\nArchivos: {sources}", schema)["sources"]
    if not result or not set(result) <= set(sources):
        raise LabError(f"Fuentes inválidas del enrutador: {result}")
    return result


def razonar(llm, question, docs, sentences):
    result = llm.ask("Produce exactamente UNA oración breve de razonamiento basada en la evidencia, "
                    "para orientar la siguiente búsqueda. No inventes hechos.\n"
                    f"Pregunta: {question}\nPasos anteriores: {sentences}\nEvidencia:\n{context(docs)}")
    return re.split(r"(?<=[.!?])\s+|\n", result.strip())[0]


def recuperar(collection, sentence, k, sources=None):
    return collection.search(sentence, k, sources)


def continuar_ircot(llm, question, docs, sentences):
    schema = {"type": "object", "properties": {"continuar": {"type": "boolean"}},
              "required": ["continuar"]}
    return llm.ask("¿Falta evidencia para responder la pregunta? continuar=true solo si falta.\n"
                   f"Pregunta: {question}\nPasos: {sentences}\nEvidencia:\n{context(docs)}",
                   schema)["continuar"]


def ircot(llm, question, collection, config, k, trace, sources=None):
    docs = collection.search(question, min(k, config["max_fragments"]), sources)
    trace.append({"stage": "initial_retrieval", "query": question, "documents": docs.copy()})
    sentences = []
    for _ in range(config["max_sentences"]):
        if len(docs) >= config["max_fragments"]:
            break
        sentence = razonar(llm, question, docs, sentences)
        sentences.append(sentence)
        found = recuperar(collection, sentence, k, sources)
        known = {d["id"] for d in docs}
        added = [d for d in found if d["id"] not in known][:config["max_fragments"] - len(docs)]
        docs.extend(added)
        trace.append({"stage": "ircot", "sentence": sentence, "retrieved": found,
                      "added": added, "total_fragments": len(docs)})
        if len(docs) >= config["max_fragments"] or len(sentences) >= config["max_sentences"]:
            break
        if not continuar_ircot(llm, question, docs, sentences):
            break
    return docs


def mrkl_ircot(llm, question, collection, config, k, trace):
    sources = enrutador(llm, question, collection)
    trace.append({"stage": "route", "sources": sources})
    return ircot(llm, question, collection, config, k, trace, sources)


def run_system(system, question, collection, config, k):
    llm, trace = LLM(config), []
    start = time.perf_counter()
    if system == "MRKL con IRCoT":
        docs = mrkl_ircot(llm, question, collection, config, k, trace)
    elif system == "IRCoT":
        docs = ircot(llm, question, collection, config, k, trace)
    elif system == "Self-Ask":
        docs, history = [], []
        schema = {"type": "object", "properties": {"done": {"type": "boolean"},
                  "question": {"type": "string"}}, "required": ["done", "question"]}
        for _ in range(config["max_subquestions"]):
            step = llm.ask("Decide si hace falta una subpregunta y genera solo la siguiente. "
                           "Usa las respuestas anteriores para construirla. done=true si ya basta.\n"
                           f"Pregunta original: {question}\nHistorial: {history}", schema)
            if step["done"] and docs:
                break
            subquestion = step["question"].strip() or question
            found = collection.search(subquestion, k)
            answer = responder(llm, subquestion, found)
            history.append({"question": subquestion, "answer": answer})
            trace.append({"stage": "subquestion", **history[-1], "documents": found})
            docs.extend(d for d in found if d["id"] not in {x["id"] for x in docs})
        # Previous subanswers affect later retrieval; final synthesis is grounded in original evidence.
    else:
        query, sources = question, None
        if system == "HyDE":
            query = llm.ask("Redacta un pasaje hipotético breve que respondería esta pregunta.\n" + question)
            trace.append({"stage": "hypothesis", "passage": query})
        elif system == "MRKL":
            sources = enrutador(llm, question, collection)
            trace.append({"stage": "route", "sources": sources})
        elif system != "RAG simple":
            raise LabError(f"Sistema desconocido: {system}")
        docs = collection.search(query, k, sources, passage=system == "HyDE")
        trace.append({"stage": "retrieval", "query": query, "documents": docs})
    answer = responder(llm, question, docs)
    return {"response": answer, "retrieved_contexts": [d["text"] for d in docs],
            "documents": docs, "trace": trace, "llm_calls": len(llm.calls),
            "input_tokens": sum(c["input_tokens"] for c in llm.calls),
            "output_tokens": sum(c["output_tokens"] for c in llm.calls),
            "seconds": time.perf_counter() - start, "calls": llm.calls}
