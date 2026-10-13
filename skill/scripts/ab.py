"""Blind A/B judging: sheets with the versions in random slots, a hidden key, and decoding the judges' votes.

    python ab.py make <ab_dir> <case>=<build_old>,<build_new>[,<refs_dir>] ... [--seed 7]
    python ab.py decode <ab_dir> <votes.json> [<votes2.json> ...]

make: for every case one image <ab_dir>/<case>.png: the concept (front + side, when refs are given) on top, then
rows "A" and "B" (3/4, front, side, face close-up). Which build is A is random per case (seeded); the key goes to
<ab_dir>/_key.json, which the judges must never open. Rebuild the old version on the CURRENT engine first, with
the same flags (fast vs full), so only the spec differs.
decode: votes files are the judges' JSON answers {"<case>": {"overall": "A"|"B"|"tie", ...}, ...}; prints per case
which version each judge preferred and the decision: keep the new version only when every judge prefers it
(a split or a tie keeps the old one).

The judge prompt (send it to a FRESH agent with no context, one per judge; for a second judge reverse the order
of the cases): see review.md "Blind judges".
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _engine as E  # noqa: E402

S = 300


def _row(build: Path):
    tiles = [E.tile(E.toy_crop(Image.open(build / f"view_{v}.png")), S) for v in ("34", "front", "side")]
    tiles.append(E.tile(E.head_crop(Image.open(build / "view_34.png")), S))
    return tiles


def make(ab_dir: Path, cases: list[str], seed: int = 7):
    ab_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key = {}
    for c in cases:
        name, paths = c.split("=", 1)
        parts = paths.split(",")
        old, new = Path(parts[0]), Path(parts[1])
        refs = Path(parts[2]) if len(parts) > 2 else None
        new_is_a = rng.random() < 0.5
        a, b = (new, old) if new_is_a else (old, new)
        key[name] = {"A": "new" if new_is_a else "old", "B": "old" if new_is_a else "new"}
        rows = []
        if refs:
            # under the matching columns: (3/4: none), front, side = the concept's left view, (face: none)
            ct = [E.tile(E.toy_crop(Image.open(refs / f)), S) if f and (refs / f).is_file() else E.tile(None, S)
                  for f in (None, "front.png", "left.png", None)]
            rows.append(("CONCEPT", ct))
        rows += [("A", _row(a)), ("B", _row(b))]
        img = E.grid(rows, S, ["3/4", "front", "side", "face 3/4"])
        ImageDraw.Draw(img).text((6, img.height - 16), f"case {name}", fill=(0, 0, 0))
        img.save(ab_dir / f"{name}.png")
    E.write_json(ab_dir / "_key.json", key)
    print(json.dumps({"cases": [str(ab_dir / f"{c.split('=')[0]}.png") for c in cases],
                      "key": str(ab_dir / "_key.json"), "note": "never show _key.json to a judge"}, indent=1))


def decode(ab_dir: Path, votes_files: list[str]):
    key = E.read_json(ab_dir / "_key.json")
    votes = [E.read_json(v) for v in votes_files]
    out = {}
    for case, k in key.items():
        prefs = []
        for v in votes:
            o = str(v.get(case, {}).get("overall", "tie")).strip().upper()
            prefs.append(k[o] if o in ("A", "B") else "tie")
        out[case] = {"judges": prefs, "decision": "new" if prefs and all(p == "new" for p in prefs) else "old"}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in ("make", "decode"):
        raise SystemExit(__doc__)
    if args[0] == "make":
        seed = 7
        if "--seed" in args:
            i = args.index("--seed")
            seed = int(args[i + 1])
            del args[i:i + 2]
        make(Path(args[1]), args[2:], seed)
    else:
        decode(Path(args[1]), args[2:])
