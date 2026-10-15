"""
Simple evaluation harness:
- Runs a set of questions against a chosen backend
- Logs latency, token estimate, and source count to JSONL
- Produces a summary report (avg latency, avg answer length)
"""
import argparse, json, statistics as stats
from pathlib import Path

def read_jsonl(path):
    items = []
    if not Path(path).exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items

def summarize(records):
    if not records:
        return {}
    lat = [r.get("latency_sec") for r in records if isinstance(r.get("latency_sec"), (int,float))]
    alen = [r.get("answer_len_chars") for r in records if isinstance(r.get("answer_len_chars"), int)]
    tok = [r.get("answer_est_tokens") for r in records if isinstance(r.get("answer_est_tokens"), int)]
    return {
        "count": len(records),
        "avg_latency_sec": round(stats.mean(lat), 3) if lat else None,
        "p95_latency_sec": round(stats.quantiles(lat, n=20)[-1], 3) if len(lat) >= 20 else None,
        "avg_answer_len_chars": round(stats.mean(alen), 1) if alen else None,
        "avg_answer_est_tokens": round(stats.mean(tok), 1) if tok else None,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", default="reports/rag_cli_metrics.jsonl")
    args = parser.parse_args()

    items = read_jsonl(args.metrics_path)
    summary = summarize(items)
    print("[EVAL] Summary:")
    print(json.dumps(summary, indent=2))

    # Write markdown report
    md = "# RAG Evaluation Report\\n\\n"
    md += "Metrics source: `"+args.metrics_path+"`\\n\\n"
    if summary:
        md += "## Summary\\n"
        for k,v in summary.items():
            md += f"- **{k}**: {v}\\n"
    else:
        md += "_No records found._\\n"

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/eval_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("[EVAL] Wrote reports/eval_report.md")

if __name__ == "__main__":
    main()
