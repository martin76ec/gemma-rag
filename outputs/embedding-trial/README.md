# Comparación de embeddings — recuperación

Cuatro modelos, ocho preguntas y top-k 2/4; los diez documentos se indexaron en cada caso. Tokenizador TinyBERT y chunker por caracteres.

| embedding                                                   |   top_k |   source_recall |   source_hit |   seconds |
|:------------------------------------------------------------|--------:|----------------:|-------------:|----------:|
| BAAI/bge-m3                                                 |       2 |          0.6042 |         1    |    0.0586 |
| BAAI/bge-m3                                                 |       4 |          0.875  |         1    |    0.0577 |
| intfloat/multilingual-e5-small                              |       2 |          0.7083 |         1    |    0.0113 |
| intfloat/multilingual-e5-small                              |       4 |          0.8125 |         1    |    0.0103 |
| lightonai/mDenseOn                                          |       2 |          0.6667 |         1    |    0.0283 |
| lightonai/mDenseOn                                          |       4 |          0.9375 |         1    |    0.0282 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |       2 |          0.5208 |         0.75 |    0.0109 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |       4 |          0.7708 |         1    |    0.01   |

Con k=4, mDenseOn obtuvo el mayor recall de archivos anotados (0.9375), seguido de BGE-M3 (0.8750). Con k=2, E5-small obtuvo el mayor recall (0.7083). Estas mediciones no son answer correctness ni faithfulness y no seleccionan la colección final: esa selección requiere RAGAS.

![Recall](../plots/embedding-trial/retrieval.png)

![Por pregunta](../plots/embedding-trial/retrieval_by_question.png)
