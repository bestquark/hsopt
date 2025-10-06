# extract_code.py
# to use run
# python checkpoint2file.py /path/to/json_or_dir -o out_py
import json
import argparse
import pathlib


def iter_json_objs(p: pathlib.Path):
    if p.is_dir():
        for f in p.rglob("*.json"):
            yield from iter_json_objs(f)
    else:
        txt = p.read_text(encoding="utf-8")
        try:
            obj = json.loads(txt)  # plain JSON
            yield obj
        except json.JSONDecodeError:
            for line in txt.splitlines():  # JSONL
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def pick_ext(lang: str | None):
    return {
        "python": "py",
        "py": "py",
        "cpp": "cpp",
        "c": "c",
        "javascript": "js",
        "typescript": "ts",
    }.get((lang or "").lower(), "py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="JSON file(s) or folder(s)")
    ap.add_argument("-o", "--outdir", default="extracted_code", help="Output directory")
    ap.add_argument("--field", default="code", help="JSON field containing code")
    ap.add_argument(
        "--name", default="{id}.{ext}", help="Filename template (fields: id, ext)"
    )
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    count = 0
    for inp in args.inputs:
        for obj in iter_json_objs(pathlib.Path(inp)):
            if not isinstance(obj, dict) or args.field not in obj:
                continue
            code = obj[args.field]
            if not isinstance(code, str):
                continue
            ext = pick_ext(obj.get("language"))
            file_id = obj.get("id") or f"no_id_{count}"
            name = args.name.format(id=file_id, ext=ext)
            path = outdir / name
            path.write_text(code, encoding="utf-8")
            count += 1
    print(f"Wrote {count} file(s) to {outdir}")


if __name__ == "__main__":
    main()
