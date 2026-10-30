"""Freeze scientific campaign records; never export sandbox messages or secrets.

This read-only exporter retains every proposed PF, original decision, aggregate
result, and compact measured-case receipt. Detailed baseline/final receipts retain compiler and accelerator provenance.
Every proposal retains its scored case metrics. Large circuit expansions and
Krylov vectors remain in the original archive.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REMOTE = r"""
import gzip,hashlib,io,json,math,sys
from pathlib import Path

CAMPAIGNS = {
    "molecular": ("elote115_abs_tight",100,115),
    "reactions": ("elote_delta_ground10_warm_a100_20261023",50,10),
}
transport=gzip.GzipFile(fileobj=sys.stdout.buffer,mode="wb",mtime=0)
sys.stdout=io.TextIOWrapper(transport,encoding="utf-8",write_through=True)
campaign,budget,count=CAMPAIGNS[CAMPAIGN]
root=Path(WORKDIR)/"runs"/campaign
sha=lambda data:hashlib.sha256(data).hexdigest()

def read(path):
    stored=path.read_bytes()
    data=gzip.decompress(stored) if path.suffix==".gz" else stored
    return json.loads(data),{"path":str(path.relative_to(root)),"stored_sha256":sha(stored),
                             "content_sha256":sha(data)}

def finite(value):
    if isinstance(value,float) and not math.isfinite(value):
        return {"nonfinite":str(value)}
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,list):return [finite(v) for v in value]
    return value

def emit(value):
    print(json.dumps(finite(value),sort_keys=True,separators=(",",":"),allow_nan=False),flush=True)

def compact(value):
    if not isinstance(value,dict):return value
    result={k:v for k,v in value.items() if
            (v is None or isinstance(v,(str,int,float,bool)))
            and not k.endswith("_path") and k not in ("task_id",)}
    for key in ("policy","rotation_synthesis","logical_circuit","spectral_solver",
                "phase_encoding","pf_peak_selection","timing_seconds"):
        if isinstance(value.get(key),dict):
            result[key]={k:v for k,v in value[key].items()
                         if v is None or isinstance(v,(str,int,float,bool))}
    if isinstance(value.get("species"),dict):
        result["species"]={name:compact(row) for name,row in value["species"].items()}
    elif isinstance(value.get("species"),list):
        result["species"]=[compact(row) for row in value["species"]]
    return result

seeds=sorted(root.glob("seed-*"))
assert len(seeds)==5
headers=[]
retained={}
suite_sha=None
for seed in seeds:
    manifest,mref=read(seed/"manifest.json")
    state,_=read(seed/"state.json")
    suite_bytes=(seed/"suite.json").read_bytes()
    assert sha(suite_bytes)==manifest["suite_sha256"]
    assert manifest["iterations_target"]==state["iterations_completed"]==budget
    assert manifest["energy_target"]=="ground_branch"
    suite=json.loads(suite_bytes)
    assert len(suite["cases"])==count
    if suite_sha is None:suite_sha=sha(suite_bytes)
    assert suite_sha==sha(suite_bytes)
    retained[seed.name]=max(i for i in range(budget+1) if read(seed/"iterations"/f"{i:04d}"/"decision.json")[0]["accepted"])
    headers.append({"seed":seed.name,"manifest":manifest,"manifest_source":mref,"retained_iteration":retained[seed.name]})
emit({"record_type":"campaign","schema":"hsopt-scientific-trajectories-v1",
      "campaign_id":campaign,"budget":budget,"case_count":count,"seed_count":5,
      "suite_sha256":suite_sha,"suite_source_utf8":suite_bytes.decode(),"seeds":headers,
      "scope":"Original candidate bytes, decisions, aggregate results and compact scientific measurements; no raw agent messages or credentials."})

for seed in seeds:
    for iteration in range(budget+1):
        folder=seed/"iterations"/f"{iteration:04d}"
        candidate=(folder/"candidate.py").read_bytes()
        decision,dref=read(folder/"decision.json")
        result,rref=read(folder/"result.json")
        requests,qref=read(folder/"requests.json")
        candidate_sha=sha(candidate)
        assert requests["shared_candidate_sha256"]==candidate_sha
        assert requests["suite_sha256"]==suite_sha
        assert decision.get("shared_candidate_sha256",candidate_sha)==candidate_sha
        assert result["case_count"]==count and len(result["case_metrics"])==count
        path=folder/"case_results.json"
        if not path.exists():path=path.with_suffix(".json.gz")
        detailed=iteration in (0,retained[seed.name])
        if detailed:
            measured,cref=read(path)
            assert isinstance(measured,dict)
            expected={row["case_id"] for row in result["case_metrics"]}
            assert set(measured)==expected
            records={name:compact(row) for name,row in measured.items()}
        else:
            measured=None
            cref={"path":str(path.relative_to(root)),"scope":"Full circuit receipt remains external; scored metrics are fingerprinted by sources.result."}
            records={row["case_id"]:row for row in result["case_metrics"]}
        if decision["accepted"]:
            assert result["passes_chemical_accuracy"] is True
            assert all(row["passes_chemical_accuracy"] for row in result["case_metrics"])
        bound=folder/"bound_policies"
        emit({"record_type":"round","seed":seed.name,"iteration":iteration,
              "candidate_source_utf8":candidate.decode(),"candidate_sha256":candidate_sha,
              "decision":decision,"result":result,"measurements":records,
              "measurement_scope":"compact_full_receipt" if detailed else "scored_case_metrics",
              "bound_policy_sha256":{p.name:sha(p.read_bytes()) for p in sorted(bound.glob("*.py"))},
              "sources":{"decision":dref,"result":rref,"requests":qref,"case_results":cref}})
        del measured,records
        if iteration%10==0:print(f"{campaign} {seed.name} {iteration}/{budget}",file=sys.stderr,flush=True)
sys.stdout.close()
"""


def export(host: str, workdir: str, campaign: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"use a fresh frozen-export directory: {output}")
    output.mkdir(parents=True)
    script = "CAMPAIGN=" + repr(campaign) + "\nWORKDIR=" + repr(workdir) + "\n" + REMOTE
    target = output / "trajectories.jsonl.gz"
    with (output / "export.log").open("w") as log:
        process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "python3", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        process.stdin.write(script.encode())
        process.stdin.close()
        count = 0
        header = None
        with (
            target.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream,
            gzip.GzipFile(fileobj=process.stdout, mode="rb") as incoming,
        ):
            for line in incoming:
                row = json.loads(line)
                if row["record_type"] == "campaign":
                    assert header is None
                    header = row
                else:
                    assert header is not None
                    source = row["candidate_source_utf8"].encode()
                    assert hashlib.sha256(source).hexdigest() == row["candidate_sha256"]
                    count += 1
                stream.write(line)
        if process.wait():
            raise RuntimeError(
                f"read-only scientific export failed; inspect {output / 'export.log'}"
            )
    assert count == header["seed_count"] * (header["budget"] + 1)
    manifest = {
        "schema": "hsopt-scientific-trajectories-export-v1",
        "campaign_id": header["campaign_id"],
        "seed_count": header["seed_count"],
        "mutation_budget_per_seed": header["budget"],
        "case_count": header["case_count"],
        "round_records": count,
        "suite_sha256": header["suite_sha256"],
        "files": {target.name: hashlib.sha256(target.read_bytes()).hexdigest()},
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh", default="omni")
    parser.add_argument("--workdir", default="/home/lmantilla/projects/autoresearch/hsopt_workdir")
    parser.add_argument("--campaign", choices=("molecular", "reactions"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.ssh, args.workdir, args.campaign, args.output), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
