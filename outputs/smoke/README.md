# Prueba de integración, no resultados del laboratorio

Pregunta técnica: ¿Cuál es la autonomía del LR-200?

Los seis sistemas se probaron con Gemma 4 E4B, E5-small, BETO cased, chunker semántico y top-k=2. Esta configuración no fue seleccionada por grid. No permite recomendar un sistema ni sustituye las ocho preguntas de clase.

| system         |   answer_correctness |   faithfulness |   llm_calls |   input_tokens |   output_tokens |   seconds |
|:---------------|---------------------:|---------------:|------------:|---------------:|----------------:|----------:|
| RAG simple     |                0.721 |              1 |           1 |            302 |              33 |     0.565 |
| HyDE           |                0.721 |              1 |           2 |            339 |             358 |     5.184 |
| MRKL           |                0.721 |              1 |           2 |            566 |              78 |     1.281 |
| Self-Ask       |                0.721 |              1 |           4 |            779 |             122 |     1.954 |
| IRCoT          |                0.721 |              1 |           3 |           1127 |              70 |     1.175 |
| MRKL con IRCoT |                0.721 |              1 |           4 |           1243 |             115 |     1.86  |
