"""Read-only mutation-history model and compact campaign viewer.

The campaign directory remains the source of truth.  This module translates
its ``campaign.json``, lane state, decisions, and archived candidate source
into one stable JSON document that both the local CLI page and a hosted
dashboard can consume.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL_FILENAME = "mutation-history.json"
PAGE_FILENAME = "index.html"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _read_agent_actions(path: Path) -> list[dict[str, Any]]:
    """Read the runner's redacted per-tool timeline without raw tool payloads."""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    actions: list[dict[str, Any]] = []
    for line in lines:
        try:
            action = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(action, dict):
            continue
        actions.append(
            {
                key: action.get(key)
                for key in ("timestamp", "tool", "target", "status", "duration_ms", "online")
            }
        )
    return actions


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the useful scalar evaluator fields without duplicating matrices."""

    return {
        key: value
        for key, value in result.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def _lane_record(campaign_root: Path, entry: dict[str, Any], default_target: int) -> dict[str, Any]:
    raw_lane_dir = Path(str(entry.get("lane_dir", "")))
    lane_dir = raw_lane_dir if raw_lane_dir.is_absolute() else campaign_root / raw_lane_dir
    manifest = _read_json(lane_dir / "manifest.json")
    state = _read_json(lane_dir / "state.json")
    target = int(manifest.get("iterations_target", default_target))
    cells: list[dict[str, Any]] = []
    accepted_iterations: list[int] = []

    for iteration in range(1, target + 1):
        iteration_dir = lane_dir / "iterations" / f"{iteration:04d}"
        decision = _read_json(iteration_dir / "decision.json")
        result = _read_json(iteration_dir / "result.json")
        candidate_source = _read_text(iteration_dir / "candidate.py")
        if decision:
            status = "accepted" if bool(decision.get("accepted")) else "rejected"
            if status == "accepted":
                accepted_iterations.append(iteration)
        else:
            status = "pending"
        cells.append(
            {
                "iteration": iteration,
                "status": status,
                "is_current_best": False,
                "has_proposal": candidate_source is not None,
                "candidate_source": candidate_source,
                "decision": decision,
                "result": _public_result(result),
                "mutation": _read_json(iteration_dir / "mutation_result.json"),
                "agent_audit": _read_json(iteration_dir / "agent_audit.json"),
                "agent_actions": _read_agent_actions(iteration_dir / "agent_actions.jsonl"),
            }
        )

    current_best_iteration = accepted_iterations[-1] if accepted_iterations else 0
    if current_best_iteration:
        cells[current_best_iteration - 1]["is_current_best"] = True
    baseline_result = _read_json(lane_dir / "baseline_result.json")
    family = str(entry.get("family") or manifest.get("family") or "unknown")
    instance = str(entry.get("instance") or manifest.get("instance") or lane_dir.parent.name)
    replicate = str(entry.get("replicate") or manifest.get("replicate") or lane_dir.name)
    return {
        "id": raw_lane_dir.as_posix() or lane_dir.name,
        "family": family,
        "instance": instance,
        "replicate": replicate,
        "label": f"{instance} · {replicate}",
        "lane_dir": str(lane_dir),
        "iterations_target": target,
        "iterations_completed": int(state.get("iterations_completed", 0)),
        "accepted_mutations": int(state.get("accepted_mutations", len(accepted_iterations))),
        "pending_kind": state.get("pending_kind"),
        "pending_iteration": state.get("pending_iteration"),
        "last_error": state.get("last_error"),
        "current_best_iteration": current_best_iteration,
        "baseline_is_current_best": current_best_iteration == 0,
        "baseline_source": _read_text(lane_dir / "initial_pf.py"),
        "baseline_result": _public_result(baseline_result),
        "incumbent_result": _public_result(_read_json(lane_dir / "incumbent_result.json")),
        "iterations": cells,
    }


def build_mutation_history(campaign_root: Path) -> dict[str, Any]:
    """Build the durable, site-neutral JSON view of one campaign."""

    campaign_root = campaign_root.expanduser().resolve()
    campaign = _read_json(campaign_root / "campaign.json")
    if not campaign:
        raise FileNotFoundError(f"campaign.json not found or invalid under {campaign_root}")
    entries = campaign.get("lanes")
    if not isinstance(entries, list):
        raise ValueError("campaign.json must contain a lanes list")
    default_target = int(campaign.get("iterations", 50))
    lanes = [
        _lane_record(campaign_root, entry, default_target)
        for entry in entries
        if isinstance(entry, dict)
    ]
    lanes.sort(key=lambda lane: (lane["family"], lane["instance"], lane["replicate"]))
    accepted = sum(cell["status"] == "accepted" for lane in lanes for cell in lane["iterations"])
    rejected = sum(cell["status"] == "rejected" for lane in lanes for cell in lane["iterations"])
    proposed = sum(cell["has_proposal"] for lane in lanes for cell in lane["iterations"])
    return {
        "schema": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "campaign_root": str(campaign_root),
        "campaign": {
            key: campaign.get(key)
            for key in ("created_at", "baseline_variant", "included_families", "iterations")
        },
        "summary": {
            "lanes": len(lanes),
            "iterations_target": sum(lane["iterations_target"] for lane in lanes),
            "proposals_archived": proposed,
            "accepted": accepted,
            "rejected": rejected,
            "pending": sum(lane["iterations_target"] for lane in lanes) - accepted - rejected,
        },
        "lanes": lanes,
    }


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HSOpt mutation history</title>
  <style>
    :root { color-scheme: dark; --bg:#0a0d12; --panel:#111722; --line:#273143;
      --text:#e8edf5; --muted:#8f9aac; --pending:#303847; --accepted:#8fc7a3;
      --rejected:#ce8d94; --best:#ffd76a; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:13px/1.45 ui-sans-serif,system-ui,sans-serif; }
    header { position:sticky; top:0; z-index:4; padding:18px 22px; border-bottom:1px solid var(--line);
      background:rgba(10,13,18,.94); backdrop-filter:blur(12px); }
    h1 { margin:0 0 4px; font-size:18px; letter-spacing:.01em; }
    .muted { color:var(--muted); } .summary { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
    .pill { padding:4px 9px; background:var(--panel); border:1px solid var(--line); border-radius:999px; }
    .toolbar { display:flex; gap:10px; margin-top:12px; }
    input,select { min-width:170px; padding:7px 9px; color:var(--text); background:#0d121a;
      border:1px solid var(--line); border-radius:6px; }
    main { display:grid; grid-template-columns:minmax(0,1fr) 420px; min-height:calc(100vh - 118px); }
    #ledger { min-width:0; padding:18px 22px 40px; overflow:auto; }
    .lane { display:grid; grid-template-columns:270px max-content; align-items:center; gap:12px;
      padding:7px 0; border-bottom:1px solid rgba(39,49,67,.45); }
    .lane-label { overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
    .lane-meta { font-size:11px; color:var(--muted); }
    .cells { display:grid; grid-template-columns:repeat(var(--iterations),24px); gap:4px; }
    .cell,.baseline { position:relative; width:24px; height:24px; padding:0; border:1px solid transparent;
      border-radius:5px; color:#10141a; font-size:10px; cursor:pointer; }
    .cell.pending { color:#aeb7c5; background:var(--pending); }
    .cell.accepted { background:var(--accepted); }
    .cell.rejected { background:var(--rejected); }
    .cell:hover,.cell:focus-visible,.baseline:hover,.baseline:focus-visible { outline:2px solid #fff; outline-offset:1px; }
    .star { position:absolute; top:-8px; right:-6px; color:var(--best); font-size:14px; text-shadow:0 1px 2px #000; }
    aside { position:sticky; top:118px; height:calc(100vh - 118px); border-left:1px solid var(--line);
      background:var(--panel); padding:18px; overflow:auto; }
    aside h2 { margin:0 0 4px; font-size:15px; } dl { display:grid; grid-template-columns:auto 1fr; gap:5px 12px; }
    dt { color:var(--muted); } dd { margin:0; overflow-wrap:anywhere; }
    pre { margin-top:14px; padding:12px; overflow:auto; background:#090d13; border:1px solid var(--line);
      border-radius:7px; font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre; }
    .actions { margin-top:14px; padding:12px; border:1px solid var(--line); border-radius:7px; background:#0b1119; }
    .actions h3 { margin:0 0 9px; font-size:12px; } .action { display:grid; grid-template-columns:74px 1fr auto;
      gap:9px; padding:7px 0; border-top:1px solid rgba(39,49,67,.6); }
    .action:first-of-type { border-top:0; } .action-tool { color:#a9d68f; font-family:ui-monospace,monospace; }
    .action-target { overflow-wrap:anywhere; } .action-meta { color:var(--muted); font-size:11px; white-space:nowrap; }
    .empty { padding:40px; color:var(--muted); text-align:center; }
    @media (max-width:900px) { main { grid-template-columns:1fr; } aside { position:relative; top:0; height:auto;
      min-height:340px; border-left:0; border-top:1px solid var(--line); } .lane { grid-template-columns:190px max-content; } }
  </style>
</head>
<body>
<header>
  <h1>Mutation history</h1><div id="stamp" class="muted"></div>
  <div id="summary" class="summary"></div>
  <div class="toolbar"><input id="search" type="search" placeholder="Filter Hamiltonians"><select id="family"><option value="">All families</option></select></div>
</header>
<main><section id="ledger"></section><aside id="detail"><div class="empty">Select an iteration to inspect its proposal.</div></aside></main>
<script id="initial-data" type="application/json">__HSOPT_DATA__</script>
<script>
const initial=JSON.parse(document.querySelector('#initial-data').textContent); let model=initial;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function actionLabel(action){
  const tool=String(action.tool||'tool'); const target=action.target?String(action.target):'';
  const verb={read:'Read file',websearch:'Web search',webfetch:'Fetched online source',codesearch:'Code search',skill:'Loaded skill',edit:'Edited file',write:'Wrote file',apply_patch:'Patched file'}[tool]||tool;
  return `${verb}${target?` · ${target}`:''}`;
}
function actionTimeline(actions){
  if(!actions?.length)return '<div class="actions"><h3>Agent actions</h3><div class="muted">No action timeline was archived for this proposal.</div></div>';
  return `<div class="actions"><h3>Agent actions · ${actions.length}</h3>${actions.map((a,i)=>`<div class="action"><span class="action-tool">${i+1}. ${esc(a.tool||'tool')}</span><span class="action-target">${esc(actionLabel(a))}</span><span class="action-meta">${esc(a.status||'unknown')}${Number.isFinite(Number(a.duration_ms))?` · ${(Number(a.duration_ms)/1000).toFixed(2)}s`:''}</span></div>`).join('')}</div>`;
}
function detail(lane,cell){
  const baseline=!cell; const d=baseline?{}:cell.decision||{}; const result=baseline?lane.baseline_result:cell.result||{};
  const source=baseline?lane.baseline_source:cell.candidate_source;
  document.querySelector('#detail').innerHTML=`<h2>${esc(lane.label)}</h2><div class="muted">${baseline?'Iteration 0 · baseline':`Iteration ${cell.iteration} · ${cell.status}`}</div>
    <dl><dt>Current best</dt><dd>${baseline?lane.baseline_is_current_best:cell.is_current_best?'yes':'no'}</dd>
    <dt>Accepted</dt><dd>${baseline?'baseline':String(d.accepted??'undecided')}</dd>
    <dt>Reason</dt><dd>${esc(d.acceptance_reason??'—')}</dd><dt>Score</dt><dd>${esc(d.candidate_score??result.score??'—')}</dd>
    <dt>T count</dt><dd>${esc(result.t_count??'—')}</dd><dt>T95</dt><dd>${esc(result.confidence_adjusted_t_count??'—')}</dd>
    <dt>Task</dt><dd>${esc(d.task_id??result.task_id??'—')}</dd></dl>${baseline?'':actionTimeline(cell.agent_actions)}<pre>${esc(source??'# No proposal has been archived for this iteration.')}</pre>`;
}
function render(){
  const q=document.querySelector('#search').value.toLowerCase(), family=document.querySelector('#family').value;
  document.querySelector('#stamp').textContent=`Read-only view · ${model.generated_at} · ${model.campaign_root}`;
  document.querySelector('#summary').innerHTML=Object.entries(model.summary).map(([k,v])=>`<span class="pill">${esc(k.replaceAll('_',' '))}: <b>${v}</b></span>`).join('');
  const lanes=model.lanes.filter(l=>(!family||l.family===family)&&(`${l.instance} ${l.replicate}`.toLowerCase().includes(q)));
  document.querySelector('#ledger').innerHTML=lanes.map((l,i)=>`<div class="lane"><div class="lane-label" title="${esc(l.label)}"><button class="baseline" data-lane="${i}" aria-label="Open baseline">0${l.baseline_is_current_best?'<span class="star">★</span>':''}</button> ${esc(l.label)}<div class="lane-meta">${esc(l.family)} · ${l.iterations_completed}/${l.iterations_target} · ${l.accepted_mutations} accepted</div></div><div class="cells" style="--iterations:${l.iterations_target}">${l.iterations.map(c=>`<button class="cell ${c.status}" data-lane="${i}" data-iteration="${c.iteration}" aria-label="Iteration ${c.iteration}: ${c.status}">${c.iteration}${c.is_current_best?'<span class="star">★</span>':''}</button>`).join('')}</div></div>`).join('')||'<div class="empty">No lanes match this filter.</div>';
  const visible=lanes;
  document.querySelectorAll('[data-lane]').forEach(b=>b.addEventListener('click',()=>{const lane=visible[+b.dataset.lane]; detail(lane,b.dataset.iteration?lane.iterations[+b.dataset.iteration-1]:null);}));
}
for(const f of [...new Set(model.lanes.map(l=>l.family))].sort()) document.querySelector('#family').insertAdjacentHTML('beforeend',`<option>${esc(f)}</option>`);
document.querySelector('#search').addEventListener('input',render); document.querySelector('#family').addEventListener('change',render); render();
setInterval(async()=>{try{const r=await fetch('./mutation-history.json?'+Date.now(),{cache:'no-store'});if(r.ok){model=await r.json();render();}}catch(_){ }},20000);
</script>
</body></html>
"""


def write_mutation_history(campaign_root: Path, output_dir: Path) -> dict[str, Any]:
    """Write the reusable JSON model and a self-contained static viewer."""

    model = build_mutation_history(campaign_root)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(model, indent=2, sort_keys=True)
    (output_dir / MODEL_FILENAME).write_text(encoded + "\n", encoding="utf-8")
    inline = json.dumps(model, separators=(",", ":")).replace("</", "<\\/")
    (output_dir / PAGE_FILENAME).write_text(
        _PAGE.replace("__HSOPT_DATA__", inline), encoding="utf-8"
    )
    return model


class _RefreshingHandler(SimpleHTTPRequestHandler):
    campaign_root: Path
    output_dir: Path

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler API
        if self.path.split("?", 1)[0].endswith(f"/{MODEL_FILENAME}"):
            write_mutation_history(self.campaign_root, self.output_dir)
        super().do_GET()


def serve_mutation_history(
    campaign_root: Path,
    output_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Serve the page while refreshing its JSON model on every poll."""

    campaign_root = campaign_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    write_mutation_history(campaign_root, output_dir)
    handler_type = type(
        "RefreshingMutationHistoryHandler",
        (_RefreshingHandler,),
        {"campaign_root": campaign_root, "output_dir": output_dir},
    )
    handler = partial(handler_type, directory=str(output_dir))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"HSOpt mutation history: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("HSOPT_VIEW_OUTPUT", "mutation-history-view")),
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.serve:
        serve_mutation_history(
            args.campaign_root,
            args.output_dir,
            host=args.host,
            port=args.port,
        )
        return 0
    model = write_mutation_history(args.campaign_root, args.output_dir)
    print(
        json.dumps(
            {
                "status": "written",
                "page": str(args.output_dir.resolve() / PAGE_FILENAME),
                "model": str(args.output_dir.resolve() / MODEL_FILENAME),
                "summary": model["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
