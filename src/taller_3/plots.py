"""Offline plots from saved measurements, keeping each experiment separate."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import OUT, ROOT, errors, read

COLORS = ["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#ec4899", "#64748b"]


def export(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_results(frame, directory, title, group="system"):
    frame = frame.copy()
    frame["total_tokens"] = frame.input_tokens + frame.output_tokens
    metrics = [m for m in ("answer_correctness", "faithfulness") if m in frame and frame[m].notna().any()]
    costs = ["input_tokens", "output_tokens", "seconds", "llm_calls", "total_tokens"]
    summary = frame.groupby(group, sort=False)[metrics + costs].mean()
    labels = [f"{label}\n(n={len(frame[frame[group] == label])})" for label in summary.index]
    height = max(4, len(summary) * .55 + 1.8)
    fig, axes = plt.subplots(1, 3, figsize=(14, height), layout="constrained")
    summary[["input_tokens", "output_tokens"]].plot.barh(
        stacked=True, ax=axes[0], color=COLORS[:2], width=.65)
    axes[0].legend(["Entrada", "Salida"], loc="lower right", fontsize=8)
    for ax, column, label in zip(axes[1:], ["seconds", "llm_calls"], ["Segundos", "Llamadas al LLM"]):
        ax.barh(range(len(summary)), summary[column], color=COLORS[0])
        ax.set_xlabel(label)
        ax.set_yticks(range(len(summary)), labels)
    axes[0].set(xlabel="Tokens por respuesta · sin juez", ylabel="")
    axes[0].set_yticklabels(labels)
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle(title + "\nCosto medio de los registros disponibles", fontsize=13)
    export(fig, directory, "costs")
    if not metrics:
        return
    fig, ax = plt.subplots(figsize=(10, height), layout="constrained")
    summary[metrics].plot.barh(ax=ax, color=COLORS[:len(metrics)], width=.7)
    ax.set(xlim=(0, 1.12), xlabel="Puntuación media RAGAS (0–1)", ylabel="", title=title)
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=9)
    ax.set_yticklabels([f"{label}\n(n evaluadas={len(frame.loc[frame[group] == label, metrics].dropna())})"
                        for label in summary.index])
    ax.invert_yaxis()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=3)
    export(fig, directory, "quality")
    if "answer_correctness" in metrics:
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for i, (label, rows) in enumerate(frame.groupby(group, sort=False)):
            valid = rows.dropna(subset=["answer_correctness"])
            if valid.empty:
                continue
            color = COLORS[i % len(COLORS)]
            ax.scatter(valid.total_tokens, valid.answer_correctness, color=color, alpha=.35, s=25)
            ax.scatter(valid.total_tokens.mean(), valid.answer_correctness.mean(), color=color,
                       marker="D", s=65, label=label)
        ax.set(xlabel="Tokens de entrada + salida · sin juez", ylabel="Answer correctness",
               ylim=(-.03, 1.06), title=title + "\nPuntos: respuestas; rombos: medias")
        ax.legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
        ax.grid(alpha=.15)
        export(fig, directory, "correctness_vs_tokens")


def plot_retrieval(frame, directory, title):
    frame = frame.copy()
    frame["model"] = frame.embedding.str.split("/").str[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, metric, label in zip(axes, ["source_recall", "source_hit"], ["Recall de archivos", "Al menos un archivo relevante"]):
        table = frame.pivot_table(index="model", columns="top_k", values=metric, aggfunc="mean")
        table.plot.barh(ax=ax, color=COLORS, width=.65)
        ax.set(xlim=(0, 1.12), xlabel=label, ylabel="")
        ax.legend(title="top-k", loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    fig.suptitle(title + "\nRecuperación de fuentes · no son métricas RAGAS", fontsize=13)
    export(fig, directory, "retrieval")
    frame["variant"] = frame.model + " · k=" + frame.top_k.astype(str)
    table = frame.pivot_table(index="variant", columns="id", values="source_recall")
    fig, ax = plt.subplots(figsize=(10, max(3, len(table) * .55)), layout="constrained")
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e5e7eb")
    chart = ax.imshow(table, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    for (row, column), value in np.ndenumerate(table.to_numpy()):
        ax.text(column, row, "—" if pd.isna(value) else f"{value:.2f}", ha="center", va="center",
                color="white" if value > .6 else "#172033", fontsize=9)
    ax.set_title(title + "\nRecall por pregunta; gris = sin medición")
    fig.colorbar(chart, ax=ax, label="Recall de archivos")
    export(fig, directory, "retrieval_by_question")


def generate():
    with errors("plots"):
        galleries = []
        for source in sorted(OUT.iterdir()):
            if not source.is_dir() or source.name == "plots":
                continue
            destination = OUT / "plots" / source.name
            if source.name == "embedding-trial" and (source / "per_question.csv").exists():
                frame = pd.read_csv(source / "per_question.csv")
                expected = read(source / "manifest.json")["config"]["embeddings"]
                observed = frame.embedding.nunique()
                title = f"Embeddings · {observed}/{len(expected)} modelos con registros · {len(frame)} mediciones"
                plot_retrieval(frame, destination, title)
            else:
                paths = sorted((source / "records").glob("*.json"))
                if source.name in ("smoke", "queries"):
                    paths = sorted(source.glob("*.json"))
                records = [dict(record, record_id=path.stem) for path in paths
                           if isinstance(record := read(path), dict) and "input_tokens" in record]
                complete = (source / "results.json").exists()
                if complete:
                    records = read(source / "results.json")
                if not records:
                    continue
                frame = pd.DataFrame(records)
                group = "record_id" if source.name == "queries" else "key" if not complete and "key" in frame else "system"
                if "variant" in frame:
                    group = "variant"
                if group == "record_id":
                    frame[group] = [f"{i + 1} · {r['system']}" for i, r in enumerate(records)]
                scope = "Prueba de integración, una pregunta" if source.name == "smoke" else "Registros disponibles, no comparación completa"
                if complete:
                    scope = "Comparación final de sistemas"
                if "variant" in frame:
                    scope = "Comparación pareada thinking on/off; no grilla RAGAS"
                title = f"{source.name} · {len(frame)} respuestas\n{scope}"
                plot_results(frame, destination, title, group)
                destination.mkdir(parents=True, exist_ok=True)
                frame.to_csv(destination / "plotted_records.csv", index=False)
            galleries.append(f"## {source.name}\n\n{title}\n\n" + "\n\n".join(
                f"![{path.stem}]({source.name}/{path.name})" for path in sorted(destination.glob("*.png"))))
        (OUT / "plots").mkdir(exist_ok=True)
        index = OUT / "plots" / "README.md"
        index.write_text("# Gráficos de resultados guardados\n\n"
                         "Cada carpeta corresponde a una ejecución separada. No se imputan métricas faltantes. "
                         "Las medias usan solo las mediciones disponibles; n indica respuestas, no repeticiones. "
                         "Las consultas manuales pueden ser preguntas diferentes y no constituyen una comparación controlada. "
                         "También se exporta cada figura como SVG.\n\n" + "\n\n".join(galleries))
        print(f"Gráficos: {index.relative_to(ROOT)}")
        return index
