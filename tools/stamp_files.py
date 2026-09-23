#!/usr/bin/env python
"""Stamp every file in the repository: what made it, what ran it, and what still refers to it.

A file is deletable when nothing made it recently, nothing has run it, and nothing refers to it --
and until now each of those was a memory exercise. This writes one record per file to
.stamps/files.jsonl, from the four sources that already know:

  git           when a file was added and last changed, and by which revision
  run ledger    .belief/artifacts/RUN-*/command.txt: which declared runs invoked it, and when
  belief.yaml   which component claims it as code, and which declared test names it
  filesystem    for generated artifacts nobody tracks, the newest mtime under them

Every stamp carries the source it came from, because they are not equally strong: "this command
invoked it at 04:44" is a fact, and "its mtime is three weeks old" is a hint. A prune list that
mixes the two silently is how live code gets deleted.

  stamp_files.py                 # write .stamps/files.jsonl and summarise
  stamp_files.py --prune         # the candidates: nothing ran it, nothing claims it, nothing names it
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOTS = ("cache", "checkpoints", "runs", "videos")
SKIP = ("third_party/", ".belief/", ".stamps/")


def _run(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=120).stdout


def tracked() -> list[str]:
    return [f for f in _run("git", "ls-files").split("\n")
            if f and not f.startswith(SKIP)]


def git_history() -> dict[str, dict]:
    """One pass over the log: when each path was first and last touched, and by what."""
    out: dict[str, dict] = {}
    log = _run("git", "log", "--reverse", "--name-only", "--format=%x00%H %ct")
    commit = when = ""
    for line in log.split("\n"):
        if line.startswith("\0"):
            commit, when = line[1:].split(" ", 1)
        elif line.strip():
            e = out.setdefault(line.strip(), {"added": when, "added_rev": commit})
            e["changed"], e["changed_rev"], e["commits"] = when, commit, e.get("commits", 0) + 1
    return out


def run_ledger() -> dict[str, list[dict]]:
    """Which declared runs named which file in their command, and when they ran."""
    invoked: dict[str, list[dict]] = {}
    for cmd_path in sorted((ROOT / ".belief/artifacts").glob("RUN-*/command.txt")):
        text = cmd_path.read_text(errors="replace")
        at = re.search(r"^at=(\S+)", text, re.M)
        run_id = cmd_path.parent.name
        for path in set(re.findall(r"(?<![\w/.])((?:tools|screwhead|scripts|tests)/[\w./-]+)", text)):
            invoked.setdefault(path, []).append({"kind": "invoked", "source": "run-ledger",
                                                 "run": run_id, "at": at.group(1) if at else None})
    return invoked


def declarations() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Which component claims each file, and which declared node names it in its text."""
    claims: dict[str, list[str]] = {}
    named: dict[str, list[str]] = {}
    belief = yaml.safe_load((ROOT / "belief.yaml").read_text())
    for comp in belief.get("components", []):
        for pattern in comp.get("code", []):
            claims.setdefault(pattern, []).append(comp["id"])
    text_nodes = [(t["id"], t.get("run", "")) for t in belief.get("tests", [])]
    consistency = yaml.safe_load((ROOT / "consistency.yaml").read_text())
    for section in ("lemmas", "branches"):
        for node in consistency.get(section, []):
            text_nodes.append((node["id"], f"{node.get('statement', '')} {node.get('derivation_rule', '')}"))
    for node_id, text in text_nodes:
        for path in set(re.findall(r"(?<![\w/.])((?:tools|screwhead|scripts|tests)/[\w./-]+)", text)):
            named.setdefault(path, []).append(node_id)
    return claims, named


def _claimed_by(path: str, claims: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for pattern, ids in claims.items():
        if path == pattern or path.startswith(pattern.rstrip("/") + "/") or fnmatch(path, pattern):
            out += ids
    return sorted(set(out))


def stamp() -> list[dict]:
    history, invoked, (claims, named) = git_history(), run_ledger(), declarations()
    records = []
    for path in tracked():
        h = history.get(path, {})
        stamps = []
        if h:
            stamps.append({"kind": "added", "source": "git", "at": _iso(h["added"]), "revision": h["added_rev"][:8]})
            stamps.append({"kind": "changed", "source": "git", "at": _iso(h["changed"]),
                           "revision": h["changed_rev"][:8], "commits": h["commits"]})
        stamps += sorted(invoked.get(path, []), key=lambda s: s["at"] or "")[-3:]
        for cid in _claimed_by(path, claims):
            stamps.append({"kind": "claimed", "source": "belief.yaml", "by": cid})
        for nid in sorted(set(named.get(path, []))):
            stamps.append({"kind": "named", "source": "declaration", "by": nid})
        records.append({"path": path, "tracked": True, "stamps": stamps})

    for root in ARTIFACT_ROOTS:                       # generated, untracked, dated by the filesystem
        base = ROOT / root
        if not base.exists():
            continue
        for entry in sorted(base.iterdir()):
            newest = max((p.stat().st_mtime for p in entry.rglob("*") if p.is_file()),
                         default=entry.stat().st_mtime) if entry.is_dir() else entry.stat().st_mtime
            size = sum(p.stat().st_size for p in entry.rglob("*") if p.is_file()) if entry.is_dir() \
                else entry.stat().st_size
            records.append({"path": f"{root}/{entry.name}", "tracked": False, "stamps": [
                {"kind": "generated", "source": "mtime", "at": _iso(newest), "bytes": size}]})
    return records


def _iso(ts) -> str:
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", action="store_true", help="list files nothing runs, claims or names")
    ap.add_argument("--out", default=".stamps/files.jsonl")
    args = ap.parse_args()

    records = stamp()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    kinds = {}
    for r in records:
        for s in r["stamps"]:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    print(json.dumps({"files": len(records), "stamps": kinds, "out": str(out)}))

    if args.prune:
        for r in records:
            if not r["tracked"] or not r["path"].endswith((".py", ".sh")):
                continue                       # documents and declarations answer to readers, not runs
            k = {s["kind"] for s in r["stamps"]}
            if not (k & {"invoked", "claimed", "named"}):
                last = next((s for s in r["stamps"] if s["kind"] == "changed"), {})
                print(json.dumps({"candidate": r["path"], "last_changed": last.get("at"),
                                  "why": "no declared run invoked it, no component claims it, "
                                         "no declaration names it"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
