"""The neural net chooses, the engine measures: Claude picks among amikit.cute's suggestions with the renders in view.

    python -m amikit.cute_llm <spec.json> [--ref concept.png] [--out DIR] [--apply]

The engine owns every number: amikit.cute measures the built toy, scores it against the baby-schema targets and
proposes concrete edit ops with their predicted effect. This module sends that report, the suggestions and the
pictures (head close-ups front / 35 deg / 90 deg, the full 3/4 view, and the concept when given) to Claude (vision)
and gets back which suggestions to apply (by number, optionally with a small numeric adjustment of a "set" value)
and why. The chosen ops are validated with edit.apply (which rebuilds the toy) and re-measured.

Without ANTHROPIC_API_KEY (or the `anthropic` package) it does nothing and says so (exit code 0, {"skipped": ...});
the agent path is `amikit_job.py cute <spec> <out>` in the amikit-toy skill, where the Claude Code session is the
chooser.
"""
from __future__ import annotations
import argparse
import base64
import copy
import json
import os
import sys
from pathlib import Path

from . import edit

MODEL = "claude-opus-5-5"

CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "suggestion": {"type": "integer", "description": "1-based number of the engine's suggestion"},
                    "value_override_json": {"type": "string",
                                            "description": "Optional: a JSON value replacing the value of the "
                                                           "suggestion's single 'set' op (e.g. a smaller eye move); "
                                                           "empty string to keep the engine's value"},
                    "reason": {"type": "string"},
                },
                "required": ["suggestion", "value_override_json", "reason"],
                "additionalProperties": False,
            },
        },
        "rejected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"suggestion": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["suggestion", "reason"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string", "description": "One or two sentences: what will look cuter / closer to the "
                                                     "concept after the chosen edits."},
    },
    "required": ["choices", "rejected", "summary"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You are the art director of a crocheted amigurumi toy engine. The engine has built the toy from
its written pattern, measured its proportions against baby-schema targets (cute = big round head, eyes on or a bit
below the head's middle and wide apart, small low muzzle, short chubby limbs, thick upright ears), and proposed
concrete edits with their predicted score change. You see the renders (and the concept picture when there is one).

Decide which suggestions to apply. Rules:
- The concept picture wins over the generic targets: if the concept clearly has a feature the targets call "not
  cute" (a capybara's big muzzle, wide-set eyes), reject that suggestion and say so.
- A face is a layout: reject a move that would put the eyes on top of brows, patches or colour markings drawn for
  the old position (look at the close-ups).
- Prefer few, clearly visible improvements; never pick two suggestions that fight each other.
- You may soften a single "set" value (e.g. move the eyes 1 round instead of 2) through value_override_json; you may
  not invent new ops or paths. The engine re-measures whatever you choose.
Answer with the JSON schema only."""


def _img_block(path):
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def build_request(report, images, spec_summary):
    """System + user content (exposed for tests): the report text, numbered suggestions, then labelled pictures."""
    from .cute import report_text
    lines = ["<engine_report>", report_text(report), "</engine_report>", "", "<suggestions_json>"]
    for i, c in enumerate(report["suggestions"], 1):
        lines.append(json.dumps({"n": i, "why": c["why"], "ops": c["ops"], "predicted": c.get("predicted")},
                                ensure_ascii=False))
    lines += ["</suggestions_json>", "", "<spec_summary>", spec_summary, "</spec_summary>"]
    content = [{"type": "text", "text": "\n".join(lines)}]
    for label, p in images:
        content.append({"type": "text", "text": label})
        content.append(_img_block(p))
    content.append({"type": "text", "text": "Choose the suggestions to apply."})
    system = [{"type": "text", "text": INSTRUCTIONS}]
    return system, [{"role": "user", "content": content}]


def _client():
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None, "no ANTHROPIC_API_KEY: nothing sent (use `amikit_job.py cute` and choose in the session)"
    try:
        import anthropic
    except ImportError:
        return None, "the 'anthropic' package is not installed"
    return anthropic.Anthropic(), None


def apply_choices(spec, report, choices):
    """Engine ops for the chosen suggestions (with value overrides), validated by edit.apply."""
    ops = []
    for ch in choices:
        i = int(ch["suggestion"]) - 1
        if not 0 <= i < len(report["suggestions"]):
            continue
        sops = copy.deepcopy(report["suggestions"][i]["ops"])
        ov = (ch.get("value_override_json") or "").strip()
        sets = [o for o in sops if o.get("op") == "set"]
        if ov and len(sets) == 1:
            sets[0]["value"] = json.loads(ov)
        ops += sops
    new, rep = edit.apply(spec, ops)
    return new, ops, rep


def choose(spec, ref=None, out=None, client=None):
    """Run the engine report, render the views, ask Claude. Returns a dict; {"skipped": reason} without a key."""
    from .toy import Toy
    from .cute import report, head_closeups, measure, score
    cl, why = (client, None) if client is not None else _client()
    toy = Toy(spec)
    rep = report(spec, toy=toy, predict=cl is not None)
    if cl is None:
        return {"skipped": why, "report": rep}
    if not rep["suggestions"]:
        return {"choices": [], "summary": "nothing to change: every measured feature is in the cute range",
                "report": rep}
    out = Path(out or Path.cwd() / "_cute_llm")
    views = head_closeups(toy, out / "closeups", ref=None)
    images = []
    if ref:
        images.append(("Concept picture (the target):", ref))
    images += [("Head close-up, front:", views["head_front"]), ("Head close-up, 35 degrees:", views["head_35"]),
               ("Head close-up, 90 degrees:", views["head_90"]), ("Full toy, 3/4 view:", views["full_34"])]
    system, messages = build_request(rep, images, edit.summary(spec))
    import anthropic
    try:
        resp = cl.beta.messages.create(
            model=MODEL, max_tokens=16000, system=system, messages=messages,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": CHOICE_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
    except anthropic.APIStatusError as ex:
        return {"skipped": f"Claude API error {ex.status_code}: {ex.message}", "report": rep}
    except anthropic.APIConnectionError:
        return {"skipped": "cannot reach the Claude API (network)", "report": rep}
    if resp.stop_reason in ("refusal", "max_tokens"):
        return {"skipped": f"no answer ({resp.stop_reason})", "report": rep}
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text)
    new, ops, er = apply_choices(spec, rep, data.get("choices", []))
    after = score(measure(Toy(new)))["total"]
    return {"choices": data.get("choices", []), "rejected": data.get("rejected", []), "summary": data.get("summary"),
            "ops": ops, "spec": new, "score_before": rep["score"]["total"], "score_after": after,
            "flags": er.get("flags", []), "report": rep}


def main(argv=None):
    for s_ in (sys.stdout, sys.stderr):
        try:
            s_.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="python -m amikit.cute_llm")
    ap.add_argument("spec"); ap.add_argument("--ref"); ap.add_argument("--out")
    ap.add_argument("--apply", action="store_true", help="write <out>/<name>_cute.json with the chosen edits")
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8-sig"))
    res = choose(spec, a.ref, a.out)
    if "skipped" in res:
        print(json.dumps({"skipped": res["skipped"], "score": res["report"]["score"]["total"],
                          "suggestions": [c["why"] for c in res["report"]["suggestions"]]}, ensure_ascii=False,
                         indent=1))
        return 0
    if a.apply and res.get("spec"):
        out = Path(a.out or Path(a.spec).parent)
        out.mkdir(parents=True, exist_ok=True)
        dst = out / (Path(a.spec).stem + "_cute.json")
        dst.write_text(json.dumps(res["spec"], ensure_ascii=False, indent=1), encoding="utf8")
        res["written"] = str(dst)
    print(json.dumps({k: v for k, v in res.items() if k not in ("report", "spec")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
