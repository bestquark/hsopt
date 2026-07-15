#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaign_common import db_connect, ensure_dirs, html_escape, init_db, now_iso, write_json


def fmt_int(value):
    if value is None:
        return ""
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def fmt_float(value, digits=3):
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render campaign telemetry HTML.")
    parser.add_argument("--campaign-root", type=Path, default=Path.cwd())
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    root = args.campaign_root.resolve()
    paths = ensure_dirs(root)
    conn = db_connect(root)
    init_db(conn)

    lanes = conn.execute(
        """
        SELECT * FROM lanes
        ORDER BY active DESC, COALESCE(best_score, 9e99), nqubits, terms, lane_id
        LIMIT ?
        """,
        (args.limit,),
    ).fetchall()
    counts = conn.execute(
        """
        SELECT
          COUNT(*) AS total,
          SUM(active) AS active,
          SUM(CASE WHEN status='mutating' THEN 1 ELSE 0 END) AS mutating,
          SUM(CASE WHEN status='evaluating' THEN 1 ELSE 0 END) AS evaluating,
          SUM(CASE WHEN status='evaluated' THEN 1 ELSE 0 END) AS evaluated
        FROM lanes
        """
    ).fetchone()
    events = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT 40").fetchall()

    status = {
        "updated_at": now_iso(),
        "counts": dict(counts) if counts else {},
        "lanes": [dict(row) for row in lanes],
        "events": [dict(row) for row in events],
    }
    write_json(paths["telemetry"] / "status.json", status)

    rows_html = []
    for row in lanes:
        config = row["current_config_json"]
        if config:
            try:
                config_text = json.dumps(json.loads(config), sort_keys=True)
            except Exception:
                config_text = config
        else:
            config_text = ""
        rows_html.append(
            "<tr>"
            f"<td>{html_escape(row['lane_id'])}</td>"
            f"<td>{html_escape(row['molecule'])}</td>"
            f"<td>{row['nqubits']}</td>"
            f"<td>{row['electrons']}</td>"
            f"<td>{fmt_int(row['terms'])}</td>"
            f"<td>{row['iteration']}</td>"
            f"<td>{html_escape(row['status'])}</td>"
            f"<td>{fmt_int(row['best_t_gates_qpe'])}</td>"
            f"<td>{fmt_float(row['best_total_error_hartree'], 4)}</td>"
            f"<td>{fmt_float(row['best_score'], 4)}</td>"
            f"<td>{fmt_float(row['estimated_state_workspace_gib'], 3)}</td>"
            f"<td>{fmt_float(row['estimated_sector_workspace_gib'], 3)}</td>"
            f"<td>{html_escape(config_text)}</td>"
            "</tr>"
        )

    event_html = "\n".join(
        f"<li><b>{html_escape(row['created_at'])}</b> {html_escape(row['kind'])}: {html_escape(row['message'])}</li>"
        for row in events
    )
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<meta http-equiv=\"refresh\" content=\"45\">
<title>HamLib A100 PF Campaign</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f4; color: #1e241f; }}
header {{ padding: 18px 24px; background: #1f3529; color: #fff; }}
h1 {{ margin: 0; font-size: 22px; font-weight: 650; }}
main {{ padding: 18px 24px 40px; }}
.metrics {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
.metric {{ background: #fff; border: 1px solid #d9dfd6; border-radius: 8px; padding: 10px 12px; min-width: 120px; }}
.metric b {{ display: block; font-size: 18px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; border: 1px solid #d9dfd6; }}
th, td {{ border-bottom: 1px solid #e5e9e1; padding: 7px 8px; font-size: 12px; text-align: left; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #eef2ea; z-index: 1; }}
td:nth-child(13) {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; max-width: 420px; overflow-wrap: anywhere; }}
.events {{ margin-top: 20px; background: #fff; border: 1px solid #d9dfd6; border-radius: 8px; padding: 12px 16px; }}
</style>
</head>
<body>
<header><h1>HamLib A100 product-formula autoresearch</h1><div>Updated {html_escape(now_iso())}</div></header>
<main>
<div class=\"metrics\">
<div class=\"metric\"><span>Total lanes</span><b>{fmt_int(counts["total"])}</b></div>
<div class=\"metric\"><span>Active A100-fit</span><b>{fmt_int(counts["active"])}</b></div>
<div class=\"metric\"><span>Mutating</span><b>{fmt_int(counts["mutating"])}</b></div>
<div class=\"metric\"><span>Evaluating</span><b>{fmt_int(counts["evaluating"])}</b></div>
<div class=\"metric\"><span>Evaluated</span><b>{fmt_int(counts["evaluated"])}</b></div>
</div>
<table>
<thead><tr><th>Lane</th><th>Mol</th><th>n</th><th>eta</th><th>Terms</th><th>Iter</th><th>Status</th><th>Best QPE T</th><th>Best error Ha</th><th>Best score</th><th>State GiB</th><th>Sector GiB</th><th>Config</th></tr></thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
<section class=\"events\"><h2>Recent events</h2><ul>{event_html}</ul></section>
</main>
</body>
</html>
"""
    (paths["telemetry"] / "index.html").write_text(html, encoding="utf-8")
    print(paths["telemetry"] / "index.html")


if __name__ == "__main__":
    main()
