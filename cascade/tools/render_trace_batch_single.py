#!/usr/bin/env python3

import argparse
import html
import json
from pathlib import Path

from verify_trace import load_trace, write_single_report


def render_summary(path: Path, title: str, rows: list[dict]) -> None:
    tr = []
    for row in rows:
        tr.append(
            "<tr>"
            f"<td>{row['entry']}</td>"
            f"<td>{row['cycles']}</td>"
            f"<td><a href=\"{html.escape(row['report'])}\">report</a></td>"
            f"<td><a href=\"{html.escape(row['yaml'])}\">yaml</a></td>"
            f"<td><a href=\"{html.escape(row['prompt'])}\">prompt</a></td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #bbb; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <table>
    <tr><th>entry</th><th>cycles</th><th>html</th><th>yaml</th><th>prompt</th></tr>
    {''.join(tr)}
  </table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render standalone reports from batch trace outputs")
    ap.add_argument("report_dir")
    ap.add_argument("--trace-name", required=True, choices=["kestrel", "llama"])
    ap.add_argument("--label")
    args = ap.parse_args()

    report_dir = Path(args.report_dir).resolve()
    trace_name = args.trace_name
    label = args.label or trace_name
    rows: list[dict] = []

    for entry_dir in sorted(p for p in report_dir.iterdir() if p.is_dir() and p.name.startswith("entry_")):
        yaml_path = entry_dir / f"{trace_name}.yaml"
        if not yaml_path.exists():
            continue
        html_path = entry_dir / f"{trace_name}.html"
        trace = load_trace(yaml_path)
        write_single_report(html_path, label, trace)
        rows.append(
            {
                "entry": entry_dir.name,
                "cycles": len(trace.get("cycles") or []),
                "report": str(html_path.relative_to(report_dir)),
                "yaml": str(yaml_path.relative_to(report_dir)),
                "prompt": str((entry_dir / "prompt.txt").relative_to(report_dir)),
            }
        )

    summary = {
        "trace_name": trace_name,
        "count": len(rows),
        "rows": rows,
    }
    (report_dir / f"summary_{trace_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_summary(report_dir / f"summary_{trace_name}.html", f"{label} standalone trace reports", rows)


if __name__ == "__main__":
    main()
