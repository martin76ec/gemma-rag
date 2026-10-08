import os
from pathlib import Path

_cache = Path(__file__).resolve().parents[2] / ".cache"
os.environ.setdefault("HF_HOME", str(_cache / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


def main() -> None:
    import argparse

    from .systems import SYSTEMS

    parser = argparse.ArgumentParser(description="Taller 3 · Lumina Robotics")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="Ejecutar la grilla y comparar sistemas")
    commands.add_parser("plots", help="Graficar resultados guardados sin inferencia")
    thinking = commands.add_parser("compare-thinking", help="Comparación pareada con/sin pensamiento")
    thinking.add_argument("--systems", nargs="+", choices=SYSTEMS, default=["RAG simple"])
    thinking.add_argument("--embedding", default="lightonai/mDenseOn")
    thinking.add_argument("--top-k", type=int, default=4)
    ask = commands.add_parser("ask", help="Consultar una pregunta sin grilla ni juez")
    ask.add_argument("question")
    ask.add_argument("--system", choices=SYSTEMS, default="RAG simple")
    ask.add_argument("--embedding", help="ID de un embedding de config.json")
    ask.add_argument("--top-k", type=int)
    args = parser.parse_args()
    if args.command == "compare-thinking":
        from .thinking import compare
        compare(args.systems, args.embedding, args.top_k)
    elif args.command == "plots":
        from .plots import generate
        generate()
    elif args.command == "ask":
        from .query import ask_question
        ask_question(args.question, args.system, args.embedding, args.top_k)
    else:
        from .experiment import run
        print(run())
