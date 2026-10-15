"""Run AthenAI's behavior evaluation set.

    python -m evals.run_behavior_evals              # deterministic layer only (no model, seconds)
    python -m evals.run_behavior_evals --live       # also score the real Qwen model (GPU recommended)
    python -m evals.run_behavior_evals --live --repeat 3 --cases 07_confident_false_premise,13_incorrect_challenge_to_correct_answer

Run from the backend directory. Reports are written to evals/reports/ as JSON and Markdown.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="also run cases against the real model")
    parser.add_argument("--repeat", type=int, default=1, help="live runs per case (sampling is nondeterministic)")
    parser.add_argument("--cases", default="", help="comma-separated case ids to run")
    parser.add_argument("--report-dir", default=str(Path(__file__).with_name("reports")))
    args = parser.parse_args()

    if args.live:
        os.environ["ATHENAI_MOCK_LLM"] = "0"
    os.environ.setdefault("ATHENAI_LOG_LEVEL", "WARNING")

    from evals import harness

    cases = harness.load_cases()
    if args.cases:
        wanted = {item.strip() for item in args.cases.split(",") if item.strip()}
        cases = [case for case in cases if case["id"] in wanted]

    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "deterministic": [], "live": []}
    for case in cases:
        failures = harness.run_deterministic(case)
        report["deterministic"].append({"id": case["id"], "title": case["title"], "passed": not failures, "failures": failures})
        print(f"[deterministic] {'PASS' if not failures else 'FAIL'} {case['id']}")
        for failure in failures:
            print(f"    - {failure}")

    if args.live:
        import main as app_main

        llm = app_main.qwen_llm
        report["model"] = llm.model_path
        for case in cases:
            for run in range(max(1, args.repeat)):
                started = time.perf_counter()
                result = harness.run_live(case, llm)
                result["run"] = run + 1
                result["seconds"] = round(time.perf_counter() - started, 1)
                report["live"].append(result)
                print(f"[live] {'PASS' if result['passed'] else 'REVIEW'} {case['id']} (run {run + 1}, {result['seconds']}s)")
                for turn in result["turns"]:
                    for problem in turn["problems"]:
                        print(f"    - {turn['prompt'][:60]!r}: {problem}")

    summary = summarize(report, cases)
    report["summary"] = summary
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "latest.md").write_text(markdown_report(report, cases), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {report_dir / 'latest.md'}")
    return 0 if summary["deterministic_failed"] == 0 else 1


def summarize(report: dict, cases: list[dict]) -> dict:
    should_help = {case["id"]: case.get("should_help") for case in cases}
    deterministic = report["deterministic"]
    summary = {
        "deterministic_passed": sum(item["passed"] for item in deterministic),
        "deterministic_failed": sum(not item["passed"] for item in deterministic),
        "deterministic_unnecessary_refusals": sum("unnecessary refusal" in failure for item in deterministic for failure in item["failures"]),
        "deterministic_missed_harm": sum("missed harmful" in failure for item in deterministic for failure in item["failures"]),
    }
    live = report["live"]
    if live:
        summary.update({
            "live_runs": len(live),
            "live_passed": sum(item["passed"] for item in live),
            "live_needs_review": sum(not item["passed"] for item in live),
            "live_over_refusals": sum(turn["over_refusal"] for item in live for turn in item["turns"]),
            "live_missed_harm": sum(
                1 for item in live for turn in item["turns"]
                if should_help.get(item["id"]) is False and turn["route"] == "model" and not turn["refused"]
            ),
        })
    return summary


def markdown_report(report: dict, cases: list[dict]) -> str:
    by_id = {case["id"]: case for case in cases}
    lines = ["# AthenAI behavior evaluation", "", f"Generated {report['generated_at']}."]
    if report.get("model"):
        lines.append(f"Live model: `{report['model']}`.")
    lines += ["", "## Summary", ""]
    lines += [f"- {key.replace('_', ' ')}: {value}" for key, value in report["summary"].items()]
    lines += ["", "## Deterministic checks", "", "| Case | Result | Notes |", "| --- | --- | --- |"]
    for item in report["deterministic"]:
        notes = "; ".join(item["failures"]) or ""
        lines.append(f"| {item['id']} | {'Pass' if item['passed'] else 'Fail'} | {notes} |")
    if report["live"]:
        lines += ["", "## Live model responses", ""]
        for item in report["live"]:
            case = by_id[item["id"]]
            lines += [f"### {item['id']}: {item['title']} (run {item['run']}): {'Pass' if item['passed'] else 'Needs review'}", ""]
            lines += ["Expected: " + "; ".join(case["expected_behavior"]), "", "Unacceptable: " + "; ".join(case["unacceptable_behavior"]), ""]
            for turn in item["turns"]:
                lines += [f"**Prompt:** {turn['prompt']}", "", f"**Route:** {turn['route']}", ""]
                lines += ["> " + line for line in (turn["response"] or "(empty)").splitlines()]
                lines.append("")
                if turn["problems"]:
                    lines += [f"- Check: {problem}" for problem in turn["problems"]] + [""]
            if item.get("comparison"):
                lines += [f"Comparison: {item['comparison']}", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
