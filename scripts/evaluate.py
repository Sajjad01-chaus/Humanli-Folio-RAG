"""Reproducible retrieval and optional real-LLM checks; never substitutes mocked answers.

python -m scripts.evaluate --live --limit 30
Outputs raw answers, references and per-case results, not just aggregate scores.
"""

import argparse
import json
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.engine import Engine
from app.errors import ApplicationError
from app.ingest import ingest_pdf
from app.models import HistoryEntry, RetrievalMode
from app.observability import configure_logging, failure
from app.store import Store

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_MODES: tuple[RetrievalMode, ...] = ("bm25", "hybrid")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--delay", type=float, default=2)
    parser.add_argument("--output", default="evaluation/results.json")
    parser.add_argument("--corpus", choices=["synthetic", "public"], default="synthetic")
    parser.add_argument("--cases", default="evaluation/cases.json")
    parser.add_argument("--ids", help="Optional comma-separated case IDs for targeted regression runs")
    parser.add_argument(
        "--basic", action="store_true", help="Disable adaptive retrieval and grounding reflection"
    )
    args = parser.parse_args()
    settings = replace(Settings(), data_dir=ROOT / "data" / f"evaluation-{args.corpus}")
    if args.basic:
        settings = replace(settings, adaptive=False, verify=False)
    if not args.live:
        settings = replace(settings, api_key="", provider="groq")
    store = Store(settings.data_dir)
    for path in sorted((ROOT / "samples" / ("public" if args.corpus == "public" else "pdfs")).glob("*.pdf")):
        ingest_pdf(store, path.name, path.read_bytes(), settings)
    engine = Engine(store, settings)
    if args.live and not engine.llm.enabled:
        raise SystemExit("Live evaluation requires a configured LLM. No results generated.")
    cases = json.loads((ROOT / args.cases).read_text(encoding="utf-8"))
    if args.ids:
        selected = set(args.ids.split(","))
        cases = [case for case in cases if case["id"] in selected]
    cases = cases[: args.limit]
    results = []
    for case in cases:
        row = {"id": case["id"], "kind": case["kind"], "question": case["question"]}
        session_id = store.create_session()["id"]
        try:
            for previous in case.get("history", []):
                if args.live:
                    engine.ask(session_id, previous)
                    time.sleep(args.delay)
            history: list[HistoryEntry] = [
                {"question": t["question"], "answer": t["response"]["answer"]}
                for t in store.history(session_id)[-4:]
            ]
            rewritten = engine.llm.rewrite(case["question"], history) if args.live else case["question"]
            expected = {tuple(item) for item in case["evidence"]}
            row["retrieval"] = {}
            for mode in RETRIEVAL_MODES:
                hits = engine.retriever.search(rewritten, top_k=5, mode=mode)
                found = {(h["name"], h["page"]) for h in hits}
                row["retrieval"][mode] = {
                    "page_recall_at_5": len(expected & found) / len(expected) if expected else None,
                    "all_expected_pages": expected <= found if expected else None,
                    "pages": sorted(found),
                }
            if args.live:
                answer = engine.ask(session_id, case["question"])
                row["response"] = answer
                text = answer["answer"].lower()
                row["expected_term_check"] = (
                    all(any(term.lower() in text for term in options) for options in case["terms"])
                    if expected
                    else None
                )
                row["abstained"] = answer["status"] == "unavailable"
                cited = {(s["name"], s["page"]) for s in answer["sources"] if s["cited"]}
                row["cited_expected_page_recall"] = (
                    len(expected & cited) / len(expected) if expected else None
                )
            print(case["id"], case["kind"], "ok", flush=True)
        except ApplicationError as exc:
            failure("evaluation.case_failed", exc)
            row["error"] = str(exc)
            results.append(row)
            print(case["id"], "ERROR", type(exc).__name__, str(exc), flush=True)
            break
        results.append(row)
        if args.live:
            time.sleep(args.delay)
    metrics = {}
    for mode in RETRIEVAL_MODES:
        for kinds, name in [
            (None, "all_answerable"),
            ({"cross_document"}, "cross_document"),
            ({"followup"}, "followup"),
        ]:
            scores = [
                r["retrieval"][mode]["page_recall_at_5"]
                for r in results
                if "retrieval" in r
                and mode in r["retrieval"]
                and r["retrieval"][mode]["page_recall_at_5"] is not None
                and (kinds is None or r["kind"] in kinds)
            ]
            if scores:
                metrics[f"{mode}_{name}_page_recall_at_5"] = round(statistics.mean(scores), 4)
    if args.live:
        successful = [r for r in results if "response" in r]
        answerable = [r for r in successful if r["expected_term_check"] is not None]
        unanswerable = [r for r in successful if r["kind"] == "unanswerable"]
        metrics.update(
            {
                "completed": len(successful),
                "errors": sum("error" in r for r in results),
                "expected_term_pass_rate": round(
                    statistics.mean(r["expected_term_check"] for r in answerable), 4
                )
                if answerable
                else None,
                "unanswerable_abstention_rate": round(
                    statistics.mean(r["abstained"] for r in unanswerable), 4
                )
                if unanswerable
                else None,
                "answerable_abstention_rate": round(statistics.mean(r["abstained"] for r in answerable), 4)
                if answerable
                else None,
                "mean_cited_expected_page_recall": round(
                    statistics.mean(r["cited_expected_page_recall"] for r in answerable), 4
                )
                if answerable
                else None,
                "median_total_ms": statistics.median(r["response"]["timings"]["total_ms"] for r in successful)
                if successful
                else None,
            }
        )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "live_llm": args.live,
        "model": settings.model if args.live else None,
        "embedding_backend": settings.embeddings,
        "embedding_model": settings.embedding_model,
        "reranker": settings.reranker_model if settings.rerank else None,
        "adaptive_retrieval": settings.adaptive,
        "grounding_reflection": settings.verify,
        "corpus": args.corpus,
        "corpus_documents": [
            {"name": d["name"], "pages": d["pages"], "chunks": d["chunks"], "sha": d["sha"]}
            for d in store.documents()
        ],
        "caveats": [
            "Expected-term matching is a diagnostic, not semantic answer correctness.",
            "Page recall is not passage-level recall or citation entailment.",
            "Verifier uses the same LLM and is not independent proof of grounding.",
            "Cold model load is included in first retrieval; later cases are warm.",
        ],
        "metrics": metrics,
        "cases": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if any("error" in r for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure("evaluation.unhandled_failure", exc)
        raise SystemExit(
            f"Evaluation stopped: {type(exc).__name__}. Check the structured logs and fix the input or configuration before retrying."
        ) from exc
