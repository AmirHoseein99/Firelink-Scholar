"""Phase 0 pilot gate report for the v3 evaluation set.

Run:
    python -m eval.pilot

Retrieves with the production retriever (``contextfromvectors``, top-10) for
the 6 pilot questions, then reports — per question and pooled:

  - every retrieved id: in v2 pool?, v2 LLM grade, human grade, in v3 seed
    pool?, seed sources, v3 judgment
  - bias exposure: retrieved ids NOT in the v2 pool
  - cycle delta: retrieved ids with no judgment anywhere
  - pool recall for human-relevant ids (grade >= 2): v2 pool vs v3 seed pool
    (q039 target: 15 recovered -> 17/17)
  - template/link-graph contribution: pool ids added by the template source
    that dense+BM25 did not have (junk control: <= ~80 per question)
  - probe agreement (binary >= 0.8) from the LLM grades of the pilot run

Gate criteria (from the plan) are evaluated and printed with PASS/FAIL; the
report is written to ``eval/v3/reports/pilot.json`` and the retrieval run to
``eval/v3/runs/pilot_contextfromvectors_k10.run.json`` (score.py-compatible).

Note: the v2 comparison inputs (``eval/v2/candidates`` +
``eval/v2/llm-qrels.json``) were deleted after the pilot; restore them from
git history to re-run this report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.agreement import collect as collect_agreement
from eval.score import score_run, unjudged_counts

PILOT_QUESTIONS = ("q002", "q005", "q013", "q023", "q025", "q039")
Q039_RECOVERY_IDS = 17
TEMPLATE_JUNK_CAP = 80
PROBE_BINARY_GATE = 0.8
BASELINE_COVERAGE_MIN = 9
GRADE_GOLD_PAGE = 2


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def retrieve_ids(fn, query: str, k: int) -> list[tuple[str, str, float]]:
    """Return [(doc_id, title, score)] from a querying.py function."""
    out: list[tuple[str, str, float]] = []
    for i, item in enumerate(fn(query, k)):
        doc, score = None, 0.0
        if isinstance(item, tuple) and len(item) == 2 and \
                hasattr(item[0], "page_content"):
            doc, score = item[0], float(item[1])
        elif hasattr(item, "page_content"):
            doc, score = item, 0.0
        if doc is None:
            continue
        meta = getattr(doc, "metadata", None) or {}
        out.append((meta.get("document_id", ""), meta.get("title", ""), score))
    return out


def load_v2(v2_candidates: Path, v2_llm_qrels: Path, qid: str) -> tuple[dict, dict]:
    pool = json.loads((v2_candidates / f"{qid}.json").read_text(encoding="utf-8"))
    v2_ids = {c["document_id"] for c in pool}
    llm = json.loads(v2_llm_qrels.read_text(encoding="utf-8"))
    return v2_ids, (llm.get(qid) or {}).get("judgments", {})


def build_report(
    questions: list[dict],
    run: dict[str, list[tuple[str, str, float]]],
    v3_pools: dict[str, list[dict]],
    v3_judgments: dict,
    human_seeds: dict,
    v2_pools: dict[str, set[str]],
    v2_judgments: dict[str, dict[str, int]],
) -> dict:
    per_question: dict[str, dict] = {}
    for q in questions:
        qid = q["id"]
        retrieved = run.get(qid, [])
        pool = v3_pools.get(qid, [])
        pool_by_id = {c["document_id"]: c for c in pool}
        human = human_seeds.get(qid, {})
        v3_j = (v3_judgments.get(qid) or {}).get("judgments", {})
        v2_ids = v2_pools.get(qid, set())
        v2_j = v2_judgments.get(qid, {})

        judged_anywhere = set(human) | set(v2_j) | set(v3_j)
        per_id = []
        for doc_id, title, score in retrieved:
            per_id.append({
                "doc_id": doc_id,
                "title": title,
                "score": score,
                "in_v2_pool": doc_id in v2_ids,
                "v2_llm_grade": v2_j.get(doc_id),
                "human_grade": human.get(doc_id),
                "in_v3_pool": doc_id in pool_by_id,
                "v3_sources": (pool_by_id.get(doc_id) or {}).get("sources", []),
                "v3_judgment": v3_j.get(doc_id),
                "judged_anywhere": doc_id in judged_anywhere,
            })

        relevant = {d for d, g in human.items() if g >= GRADE_GOLD_PAGE}
        template_only = [
            c for c in pool
            if "template" in c.get("sources", [])
            and "dense" not in c.get("sources", [])
            and "bm25" not in c.get("sources", [])
        ]
        summary = {
            "retrieved_count": len(retrieved),
            "n_not_in_v2_pool": sum(1 for r in per_id if not r["in_v2_pool"]),
            "not_in_v2_pool": [
                {"doc_id": r["doc_id"], "title": r["title"]}
                for r in per_id if not r["in_v2_pool"]
            ],
            "n_not_judged_anywhere": sum(
                1 for r in per_id if not r["judged_anywhere"]),
            "not_judged_anywhere": [
                {"doc_id": r["doc_id"], "title": r["title"]}
                for r in per_id if not r["judged_anywhere"]
            ],
            "human_relevant_total": len(relevant),
            "v2_pool_relevant_recall": (
                len(relevant & v2_ids) / len(relevant) if relevant else None),
            "v3_seed_relevant_recall": (
                len(relevant & set(pool_by_id)) / len(relevant)
                if relevant else None),
            "template_contribution": len(template_only),
            "template_contribution_titles": sorted(
                {c["title"] for c in template_only}),
            "pool_size_v3": len(pool),
            "pool_size_v2": len(v2_ids),
        }
        per_question[qid] = {"question": q["question"], "retrieved": per_id,
                             "summary": summary}

    agreement = collect_agreement(v3_judgments, set(per_question))
    pooled_binary = agreement["pooled"]["binary"]

    human_preserved = all(
        {
            doc_id: (v3_judgments.get(qid) or {})
            .get("judgments", {}).get(doc_id)
            for doc_id in human_seeds.get(qid, {})
        }
        == human_seeds.get(qid, {})
        for qid in per_question
    )

    gates = evaluate_gates(per_question, pooled_binary, human_preserved)
    return {
        "generated_at": now_iso(),
        "retriever": "contextfromvectors",
        "k": 10,
        "per_question": per_question,
        "probe_agreement": agreement,
        "gates": gates,
    }


def evaluate_gates(
    per_question: dict,
    pooled_binary: float | None,
    human_preserved: bool,
) -> dict:
    q039 = per_question.get("q039", {}).get("summary", {})
    q039_recall = q039.get("v3_seed_relevant_recall")

    coverage = {
        qid: d["summary"]["retrieved_count"]
        - sum(1 for r in d["retrieved"] if r["v2_llm_grade"] is None)
        for qid, d in per_question.items()
    }

    template_additions = {
        qid: d["summary"]["template_contribution"]
        for qid, d in per_question.items()
    }

    return {
        "q039_recovery": {
            "pass": q039_recall == 1.0,
            "detail": f"q039 v3 seed recall of human-relevant ids = "
                      f"{q039_recall!r} (target {Q039_RECOVERY_IDS}/"
                      f"{Q039_RECOVERY_IDS} = 1.0)",
        },
        "baseline_coverage": {
            "pass": bool(coverage) and min(coverage.values())
            >= BASELINE_COVERAGE_MIN,
            "detail": f"v2-judged ids among top-10 per question: {coverage} "
                      f"(min required {BASELINE_COVERAGE_MIN})",
        },
        "template_pool_sanity": {
            "pass": all(v <= TEMPLATE_JUNK_CAP for v in template_additions.values()),
            "detail": f"template-only additions per question: "
                      f"{template_additions} (cap {TEMPLATE_JUNK_CAP}; "
                      f"eyeball the title lists for plausibility)",
        },
        "grade_sanity": {
            "pass": pooled_binary is not None
            and pooled_binary >= PROBE_BINARY_GATE,
            "detail": f"pooled probe binary agreement = {pooled_binary!r} "
                      f"(gate {PROBE_BINARY_GATE}; plus manual eyeball of "
                      f"retrieved chunks)",
        },
        "no_regression": {
            "pass": human_preserved,
            "detail": "every human grade for the pilot questions is kept "
                      "verbatim in v3 judgments (human wins)",
        },
    }


def print_report(report: dict) -> None:
    print("=== Pilot gate report (contextfromvectors, top-10) ===\n")
    for qid, d in report["per_question"].items():
        s = d["summary"]
        print(f"[{qid}] {d['question']}")
        print(f"  retrieved {s['retrieved_count']} | outside v2 pool: "
              f"{s['n_not_in_v2_pool']} | unjudged anywhere: "
              f"{s['n_not_judged_anywhere']}")
        print(f"  human-relevant recall: v2 pool "
              f"{_fmt_recall(s['v2_pool_relevant_recall'])} -> v3 seed "
              f"{_fmt_recall(s['v3_seed_relevant_recall'])} "
              f"(of {s['human_relevant_total']})")
        print(f"  template-only additions: {s['template_contribution']} | "
              f"pool size v2 {s['pool_size_v2']} -> v3 {s['pool_size_v3']}")
        for r in d["retrieved"]:
            marks = []
            if not r["in_v2_pool"]:
                marks.append("NEW-to-v2")
            if r["human_grade"] is not None:
                marks.append(f"human={r['human_grade']}")
            if r["v2_llm_grade"] is not None:
                marks.append(f"v2llm={r['v2_llm_grade']}")
            if r["v3_judgment"] is not None:
                marks.append(f"v3={r['v3_judgment']}")
            print(f"    {r['doc_id']} {r['title'][:52]!r} "
                  f"[{','.join(r['v3_sources'])}] "
                  f"{' '.join(marks)}")
        print()

    pooled = report["probe_agreement"]["pooled"]
    print(f"Probe agreement (pooled): pairs={pooled['n_pairs']} "
          f"exact={_fmt_recall(pooled['exact'])} "
          f"binary={_fmt_recall(pooled['binary'])}")

    print("\n=== GATE ===")
    for name, gate in report["gates"].items():
        print(f"  [{'PASS' if gate['pass'] else 'FAIL'}] {name}: "
              f"{gate['detail']}")
    all_pass = all(g["pass"] for g in report["gates"].values())
    print(f"\n  => {'GO (pending your sign-off)' if all_pass else 'NO-GO'} "
          f"— inspect eval/v3/reports/pilot.json")


def _fmt_recall(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path,
                    default=REPO_ROOT / "eval" / "questions.json")
    ap.add_argument("--only", type=str,
                    default=",".join(PILOT_QUESTIONS),
                    help="comma-separated pilot question ids")
    ap.add_argument("--candidates", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "candidates")
    ap.add_argument("--judgments", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "judgments.json")
    ap.add_argument("--human-seeds", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "seeds"
                    / "human-judgments.json")
    ap.add_argument("--v2-candidates", type=Path,
                    default=REPO_ROOT / "eval" / "v2" / "candidates")
    ap.add_argument("--v2-llm-qrels", type=Path,
                    default=REPO_ROOT / "eval" / "v2" / "llm-qrels.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--run-out", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "runs"
                    / "pilot_contextfromvectors_k10.run.json")
    ap.add_argument("--report", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "reports" / "pilot.json")
    args = ap.parse_args()

    only = {p.strip() for p in args.only.split(",") if p.strip()}
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [q for q in all_questions if q["id"] in only]
    if not questions:
        sys.exit(f"no pilot questions matched {sorted(only)}")

    human_seeds = {
        qid: entry["judgments"]
        for qid, entry in
        json.loads(args.human_seeds.read_text(encoding="utf-8")).items()
    }
    v3_judgments = json.loads(args.judgments.read_text(encoding="utf-8")) \
        if args.judgments.exists() else {}
    v3_pools = {
        q["id"]: json.loads(
            (args.candidates / f"{q['id']}.json").read_text(encoding="utf-8"))
        for q in questions if (args.candidates / f"{q['id']}.json").exists()
    }
    v2_pools: dict[str, set[str]] = {}
    v2_judgments: dict[str, dict[str, int]] = {}
    for q in questions:
        v2_pools[q["id"]], v2_judgments[q["id"]] = load_v2(
            args.v2_candidates, args.v2_llm_qrels, q["id"])

    print("Loading production retriever (contextfromvectors)...")
    from src.retriever import querying

    fn = querying.contextfromvectors
    run: dict[str, list[tuple[str, str, float]]] = {}
    for q in questions:
        print(f"  retrieving {q['id']}...")
        run[q["id"]] = retrieve_ids(fn, q["question"], args.k)

    report = build_report(questions, run, v3_pools, v3_judgments,
                          human_seeds, v2_pools, v2_judgments)

    args.run_out.parent.mkdir(parents=True, exist_ok=True)
    qids = [q["id"] for q in questions]
    qrels = {qid: v3_judgments[qid] for qid in qids if qid in v3_judgments}
    ks = sorted({5, args.k})
    run_ids = {qid: [r[0] for r in run[qid]] for qid in qids}
    metrics = score_run(run_ids, qrels, ks, 2)
    unjudged = unjudged_counts(run_ids, qrels)
    args.run_out.write_text(json.dumps({
        "config": {
            "fn": "contextfromvectors",
            "k": args.k,
            "min_grade": 2,
            "qrels": str(args.judgments.relative_to(REPO_ROOT)),
            "created_at": now_iso(),
            "pilot": True,
        },
        "diagnostics": {},
        "run": run_ids,
        "n_unjudged": unjudged,
        "n_unjudged_total": sum(unjudged.values()),
        "metrics": metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print_report(report)
    print(f"\nWrote run file to {args.run_out}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
