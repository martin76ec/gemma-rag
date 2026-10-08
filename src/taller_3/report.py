"""Tables and plot from measured values only."""

import pandas as pd

from .core import ROOT, save
from .plots import plot_results

METRICS = ["answer_correctness", "faithfulness", "llm_calls", "input_tokens", "output_tokens", "seconds"]


def build_report(run_dir, records, best, config):
    frame = pd.DataFrame(records)
    frame["total_tokens"] = frame.input_tokens + frame.output_tokens
    columns = ["system", "id", "type", "question", "response", *METRICS, "total_tokens"]
    frame[columns].to_csv(run_dir / "per_question.csv", index=False)
    summary = frame.groupby("system")[METRICS + ["total_tokens"]].mean().sort_values(
        ["answer_correctness", "total_tokens"], ascending=[False, True])
    by_type = frame.groupby(["type", "system"])[METRICS].mean()
    summary.to_csv(run_dir / "summary.csv")
    by_type.to_csv(run_dir / "by_type.csv")
    plot_results(frame, run_dir, f"Comparación final · {run_dir.name}")
    # Preserve full trace evidence for inspection; do not invent a causal diagnosis.
    cases = frame.sort_values("answer_correctness").drop_duplicates("system").head(3)
    save(run_dir / "failure_cases.json", cases.to_dict("records"))
    failure_text = "\n\n".join(
        f"### {row['system']} · {row['id']}\n\nPregunta: {row['question']}\n\n"
        f"Respuesta: {row['response']}\n\nReferencia: {row['reference']}\n\n"
        f"Correctness: {row['answer_correctness']:.3f}; faithfulness: {row['faithfulness']:.3f}. "
        f"Traza completa: `records/{row['key']}__{row['id']}.json`."
        for _, row in cases.iterrows())
    winner = summary.iloc[0]
    report = f"""## Resultados de la ejecución `{run_dir.name}`

Configuración seleccionada por mayor answer correctness medio: `{best}`.
La búsqueda evaluó las ocho preguntas en cada configuración y todos los diez archivos.
Todos los sistemas usan la misma colección y generador `{config['model']}` a temperatura {config['temperature']}.

### Desempeño y costo medio por pregunta

{summary.round(3).to_markdown()}

### Desempeño por tipo

{by_type.round(3).to_markdown()}

![Correctness frente a tokens](outputs/{run_dir.name}/correctness_vs_tokens.png)

![Calidad por sistema](outputs/{run_dir.name}/quality.png)

![Costos por sistema](outputs/{run_dir.name}/costs.png)

### Casos de menor puntuación para inspección

Los siguientes son casos medidos, no una afirmación automática de que todos sean fallos.
La explicación causal se debe contrastar con sus trazas.

{failure_text}

### Conclusión

Con el criterio predefinido de mayor correctness (desempate por menos tokens), se recomienda
**{summary.index[0]}** para estas preguntas: correctness {winner.answer_correctness:.3f},
faithfulness {winner.faithfulness:.3f}, {winner.llm_calls:.2f} llamadas,
{winner.total_tokens:.1f} tokens y {winner.seconds:.2f} segundos por pregunta.
No se estiman dólares: la inferencia es local y no se midió consumo eléctrico.

La selección y comparación reutilizan ocho preguntas: el resultado es descriptivo y optimista,
no una estimación de generalización. El juez LLM puede equivocarse y comparte familia con el
generador. No hay repeticiones ni intervalos de confianza; la latencia depende del hardware,
cachés y calentamiento. Los tamaños de tokenización no representan el costo de Gemma:
los tokens de costo proceden de Ollama. El costo del juez se excluye.
"""
    (run_dir / "report.md").write_text(report)
    readme = ROOT / "README.md"
    original = readme.read_text().split("<!-- RESULTS -->")[0]
    readme.write_text(original + "<!-- RESULTS -->\n\n" + report)
