"""Evaluate the system on a reference set of drug pairs (eval/pairs.csv).

Two modes:
  retrieval  No LLM, free: RxNorm + RxClass + DailyMed + PubMed only. Does the
             system find evidence for known interactions (sensitivity) and stay
             silent for pairs without one (specificity)?
  full       The whole pipeline with the LLM (Groq free tier: ~1 pair/minute).
             Does the final answer report an interaction, and how many critic
             findings are there before / after the revision round?

Results are appended to eval/results/<mode>.jsonl (one line per pair, so an
interrupted run resumes where it stopped) and summarized in eval/results/<mode>.md.

    python -m eval.run_eval retrieval
    python -m eval.run_eval full --limit 10          # first 10 pairs of each class
    python -m eval.run_eval retrieval --summary-only

The reference set is compiled by hand from well-known clinically significant
interactions (and pairs with no expected interaction); it is not taken from
DailyMed or PubMed, so the check is not circular. It is small (40 pairs).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
PAIRS = ROOT / "pairs.csv"
RESULTS = ROOT / "results"


def load_pairs() -> list[dict]:
    with PAIRS.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(p: dict) -> str:
    return f"{p['drug_a']}+{p['drug_b']}"


# --------------------------------------------------------------------------- runners

def run_retrieval(p: dict) -> dict:
    from ddi.agents import retrievers
    from ddi.sources import dailymed, pubmed, rxnorm

    t = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.map(rxnorm.normalize, (p["drug_a"], p["drug_b"]))
    out = {"names": [[a.input, a.name, a.status], [b.input, b.name, b.status]],
           "class_errors": [d.class_error for d in (a, b) if d.class_error]}
    if a.status in ("not_found", "combination") or b.status in ("not_found", "combination"):
        return {**out, "stopped": True, "seconds": round(time.time() - t, 1)}

    terms = {a.name: a.class_terms, b.name: b.class_terms}
    dm_ev, dm_err = dailymed.find_interactions(
        a.name, b.name, per_direction=retrievers.DAILYMED_PER_DIRECTION, terms=terms)
    pm_ev, pm_err = pubmed.find_interactions(a.name, b.name, keep=retrievers.PUBMED_KEEP)
    return {**out, "stopped": False, "seconds": round(time.time() - t, 1),
            "dailymed": [{"label_drug": e.meta["label_drug"], "kind": e.meta.get("kind"),
                          "match": e.meta.get("match"), "term": e.meta.get("matched_term"),
                          "section": e.meta.get("section", "")[:80]} for e in dm_ev],
            "pubmed": [{"pmid": e.meta["pmid"], "score": e.meta.get("score"),
                        "year": e.meta.get("year")} for e in pm_ev],
            "errors": dm_err + pm_err}


def run_full(p: dict) -> dict:
    from ddi import llm
    from ddi.agents.common import is_not_found_answer
    from ddi.agents.orchestrator import run

    start = len(llm.usage_log)
    t = time.time()
    rep = run(p["drug_a"], p["drug_b"])
    calls = llm.usage_log[start:]
    draft_issues = rep.draft_issues if rep.draft else rep.issues
    return {"stopped": rep.stopped, "seconds": round(time.time() - t, 1),
            "found": not rep.stopped and not is_not_found_answer(rep.answer),
            "evidence": {r.source: len(r.evidence) for r in rep.reports},
            "issues_draft": len(draft_issues), "issues_final": len(rep.issues),
            "revised": rep.revised, "revision_note": rep.revision_note,
            "llm_calls": len(calls),
            "tokens": {m: sum(c["prompt"] + c["completion"] for c in calls if c["model"] == m)
                       for m in {c["model"] for c in calls}},
            "errors": rep.errors + [e for r in rep.reports for e in r.errors],
            "answer": rep.answer}


# --------------------------------------------------------------------------- summaries

def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "—"


def summarize_retrieval(rows: list[dict]) -> str:
    pos = [r for r in rows if r["expected"] == "interaction"]
    neg = [r for r in rows if r["expected"] == "none"]
    dm = lambda r: bool(r.get("dailymed"))
    pm = lambda r: bool(r.get("pubmed"))
    any_ = lambda r: dm(r) or pm(r)
    both_dirs = lambda r: len({e["label_drug"] for e in r.get("dailymed", [])}) == 2
    by_name = lambda r: any(e["match"] == "name" for e in r.get("dailymed", []))
    critical = lambda r: any(e["kind"] in ("boxed_warning", "contraindications")
                             for e in r.get("dailymed", []))
    only_class = [r for r in pos if dm(r) and not by_name(r)]
    # network failures only; "no Drug Interactions section" (OTC labels) is not one
    network = lambda r: any("не удал" in e for e in r.get("errors", [])) or r.get("class_errors")
    with_errors = [r for r in rows if network(r)]
    no_sections = [r for r in rows if any("нет разделов" in e for e in r.get("errors", []))]
    secs = [r["seconds"] for r in rows if "seconds" in r]
    missed = [r for r in pos if not any_(r)]
    false_alarms = [r for r in neg if any_(r)]

    lines = [
        "# Оценка поиска (без LLM)", "",
        f"Пар: {len(rows)} (с взаимодействием: {len(pos)}, без: {len(neg)}).", "",
        "| Метрика | Значение |", "|---|---|",
        f"| Чувствительность, DailyMed | {pct(sum(map(dm, pos)), len(pos))} |",
        f"| Чувствительность, PubMed | {pct(sum(map(pm, pos)), len(pos))} |",
        f"| **Чувствительность, хотя бы один источник** | **{pct(sum(map(any_, pos)), len(pos))}** |",
        f"| DailyMed: найдено в обеих инструкциях | {pct(sum(map(both_dirs, pos)), len(pos))} |",
        f"| DailyMed: есть фрагмент из Boxed Warning / Contraindications | {pct(sum(map(critical, pos)), len(pos))} |",
        f"| DailyMed: найдено только по классу (RxClass / синонимы) | {pct(len(only_class), len(pos))} |",
        f"| Пары без взаимодействия: источники ничего не вернули | {pct(len(neg) - len(false_alarms), len(neg))} |",
        f"| Пары с сетевыми ошибками | {len(with_errors)} |",
        f"| Пары, где у одной инструкции нет нужных разделов (OTC) | {len(no_sections)} |",
        f"| Время на пару, медиана | {statistics.median(secs):.0f} с |" if secs else "",
        "",
    ]
    if missed:
        lines += ["**Пропущенные взаимодействия:** " +
                  ", ".join(f"{r['key']} ({r['note']})" for r in missed), ""]
    if false_alarms:
        lines += ["**Пары без взаимодействия, где источники что-то вернули** "
                  "(финальный ответ решает LLM-агент, см. режим full): " +
                  ", ".join(f"{r['key']} (DailyMed {len(r.get('dailymed', []))}, "
                            f"PubMed {len(r.get('pubmed', []))})" for r in false_alarms), ""]
    if only_class:
        lines += ["**Найдено только по классу:** " +
                  ", ".join(f"{r['key']} ({', '.join(sorted({e['term'] for e in r['dailymed']}))})"
                            for r in only_class), ""]
    return "\n".join(lines)


def summarize_full(rows: list[dict]) -> str:
    ok = [r for r in rows if not r.get("stopped")]
    pos = [r for r in ok if r["expected"] == "interaction"]
    neg = [r for r in ok if r["expected"] == "none"]
    revised = [r for r in ok if r.get("issues_draft")]
    tokens = Counter()
    for r in ok:
        tokens.update(r.get("tokens", {}))
    mean = lambda xs: f"{statistics.mean(xs):.1f}" if xs else "—"
    lines = [
        "# Оценка полного пайплайна (с LLM)", "",
        f"Пар: {len(rows)} (с взаимодействием: {len(pos)}, без: {len(neg)}).", "",
        "| Метрика | Значение |", "|---|---|",
        f"| **Взаимодействие сообщено (пары с взаимодействием)** | **{pct(sum(r['found'] for r in pos), len(pos))}** |",
        f"| «Не найдено» (пары без взаимодействия) | {pct(sum(not r['found'] for r in neg), len(neg))} |",
        f"| Замечаний критика к черновику, в среднем | {mean([r['issues_draft'] for r in ok if r['found']])} |",
        f"| Замечаний после исправления, в среднем | {mean([r['issues_final'] for r in ok if r['found']])} |",
        f"| Ответов с замечаниями → исправлено | {pct(sum(r['revised'] for r in revised), len(revised))} |",
        f"| Ответов без замечаний после исправления | {pct(sum(r['issues_final'] == 0 for r in ok if r['found']), sum(r['found'] for r in ok))} |",
        f"| Вызовов LLM на запрос, в среднем | {mean([r['llm_calls'] for r in ok])} |",
        f"| Время на запрос, медиана | {statistics.median([r['seconds'] for r in ok]):.0f} с |" if ok else "",
        f"| Токенов на запрос, в среднем | " + ", ".join(
            f"{m.split('/')[-1]}: {tokens[m] // max(len(ok), 1)}" for m in tokens) + " |",
        "",
    ]
    missed = [r for r in pos if not r["found"]]
    alarms = [r for r in neg if r["found"]]
    if missed:
        lines += ["**Не сообщено:** " + ", ".join(r["key"] for r in missed), ""]
    if alarms:
        lines += ["**Ложные сообщения о взаимодействии:** " + ", ".join(r["key"] for r in alarms), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["retrieval", "full"])
    ap.add_argument("--limit", type=int, default=0, help="first N pairs of each class")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="re-run pairs that had errors")
    ap.add_argument("--rerun", nargs="*", default=[], help="pair keys to run again, e.g. a+b")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.mode}.jsonl"
    done: dict[str, dict] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["key"]] = row

    pairs = load_pairs()
    if args.limit:
        pairs = ([p for p in pairs if p["expected"] == "interaction"][:args.limit]
                 + [p for p in pairs if p["expected"] == "none"][:args.limit])

    if not args.summary_only:
        runner = run_retrieval if args.mode == "retrieval" else run_full
        for i, p in enumerate(pairs, 1):
            k = key(p)
            prev = done.get(k)
            if prev and k not in args.rerun and \
                    not (args.retry_errors and (prev.get("errors") or prev.get("class_errors"))):
                continue
            try:
                result = runner(p)
            except Exception as exc:                       # keep going, record the failure
                result = {"crash": f"{type(exc).__name__}: {exc}", "errors": [str(exc)]}
            row = {"key": k, **p, **result}
            done[k] = row
            print(f"[{i}/{len(pairs)}] {k}: "
                  + (f"DM {len(row.get('dailymed', []))}, PM {len(row.get('pubmed', []))}"
                     if args.mode == "retrieval" else
                     f"found={row.get('found')} issues {row.get('issues_draft')}->{row.get('issues_final')}")
                  + (f" ERR {row['errors'][:1]}" if row.get("errors") else ""), flush=True)
            out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in done.values()) + "\n",
                           encoding="utf-8")

    wanted = {key(p) for p in pairs}
    rows = [r for k, r in done.items() if k in wanted and "crash" not in r]
    summary = summarize_retrieval(rows) if args.mode == "retrieval" else summarize_full(rows)
    (RESULTS / f"{args.mode}.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
