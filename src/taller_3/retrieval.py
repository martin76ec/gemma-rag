"""Two chunkers, independent HF tokenizers, and exact cosine retrieval."""

import itertools
import re
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .core import ROOT, LabError, read


@lru_cache(maxsize=2)
def encoder(name):
    return SentenceTransformer(name, device="cpu", revision=read(ROOT / "config.json")["hf_revisions"][name])


@lru_cache(maxsize=2)
def tokenizer(name):
    return AutoTokenizer.from_pretrained(name, use_fast=True,
                                        revision=read(ROOT / "config.json")["hf_revisions"][name])


def vectors(name, texts, query=False):
    prefix = ("query: " if query else "passage: ") if "e5" in name else ""
    options = {"prompt_name": "query" if query else "document"} if name == "lightonai/mDenseOn" else {}
    return encoder(name).encode([prefix + text for text in texts], normalize_embeddings=True,
                                show_progress_bar=False, **options)


def token_split(text, tok, limit):
    """Slice the original string using offsets; preserve accents and punctuation."""
    offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    boundaries = [0] + [offsets[i][0] for i in range(limit, len(offsets), limit)] + [len(text)]
    return [text[a:b] for a, b in itertools.pairwise(boundaries) if text[a:b].strip()]


def chunks(text, config, embedding, tokenizer_name, kind):
    tok = tokenizer(tokenizer_name)
    if kind == "character":
        pieces = [text[i:i + config["chunk_chars"]] for i in range(0, len(text), config["chunk_chars"])]
    else:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        vec = vectors(embedding, sentences)
        pieces, current = [], sentences[0]
        for i, sentence in enumerate(sentences[1:], 1):
            combined = current + "\n" + sentence
            if (float(vec[i - 1] @ vec[i]) < config["semantic_threshold"] or
                    len(tok.encode(combined, add_special_tokens=False)) > config["chunk_tokens"]):
                pieces.append(current)
                current = sentence
            else:
                current = combined
        pieces.append(current)
    pieces = [part for piece in pieces for part in token_split(piece, tok, config["chunk_tokens"])]
    # The embedding model keeps its own pretrained tokenizer. Avoid silent truncation.
    model = encoder(embedding)
    limit = model.max_seq_length - 8
    return [part for piece in pieces for part in token_split(piece, model.tokenizer, limit)]


class Collection:
    def __init__(self, config, embedding, tokenizer_name, kind):
        self.embedding = embedding
        self.documents = []
        paths = sorted((ROOT / "data").glob("*.txt"))
        if len(paths) != 10:
            raise LabError(f"Se esperaban diez documentos; encontrados: {len(paths)}")
        for path in paths:
            for i, text in enumerate(chunks(path.read_text(), config, embedding, tokenizer_name, kind)):
                self.documents.append({"id": f"{path.stem}:{i}", "source": path.name, "text": text})
        self.matrix = vectors(embedding, [d["text"] for d in self.documents])

    def search(self, query, k, sources=None, passage=False):
        ids = [i for i, d in enumerate(self.documents) if sources is None or d["source"] in sources]
        if not ids:
            return []
        scores = self.matrix[ids] @ vectors(self.embedding, [query], query=not passage)[0]
        order = np.argsort(-scores, kind="stable")[:k]
        return [{**self.documents[ids[i]], "score": float(scores[i])} for i in order]
