"""A toy of the user's own pet: photo -> Claude (vision) -> customisation ops -> spec -> build.

    python -m amikit.pet_llm photo.jpg <out dir> [--base dog|cat|<spec.json>] [--notes "пятно под левым глазом"]
    python -m amikit.pet_llm --ops ops.json <out dir> [--base dog]      # ops written by an agent that looked itself

The model looks at the photo and describes the pet in the engine's customisation ops (amikit/custom.py): main
coat colour first, then ears, markings, spots, eyes, nose, tail, proportions. The ops are applied by edit.apply,
which validates them by building the toy; a failure goes back to the model once. Everything the ops do is
crochet (colour changes in the rounds, embroidery, felt, other rounds / ear constructions), so the written pattern
makes exactly this toy.

Needs `anthropic` + ANTHROPIC_API_KEY for the photo step (not for --ops).
"""
from __future__ import annotations
import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path

from . import edit, custom
from .llm_edit import LLMError, LLMUnavailable, _client, MODEL

HERE = Path(__file__).resolve().parent
BASES = {"dog": HERE / "toys" / "pets" / "dog.json", "cat": HERE / "toys" / "pets" / "cat.json"}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "seen": {"type": "string", "description": "What the photo shows that matters for the toy: species, coat "
                 "colours, every marking with its side (the pet's own left/right), ear and tail kind, eye colour, "
                 "anything unusual. Short notes."},
        "explanation": {"type": "string", "description": "One or two sentences for the owner, in their language."},
        "ops_json": {"type": "string", "description": "A JSON array of customisation ops (as a string)."},
    },
    "required": ["seen", "explanation", "ops_json"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You turn a photo of someone's pet into a crocheted toy of that pet. The toy starts from a cute
base toy (summary below); you describe the pet with customisation ops and the engine rebuilds the toy, its 3D view
and its exact crochet pattern from them.

Work in this order and keep it simple (a toy, not a portrait):
1. {"op": "colour", "target": "all", "colour": <main coat colour>} - the colour most of the coat has.
2. Ears: kind (pointed = upright, floppy = hanging hound ears, folded = button / rose ears that fold forward,
   round, long) only if they differ from the base; ear colour if it differs from the coat; inner colour for
   upright ears.
3. Big markings: muzzle, blaze, mask, chest / bib, belly, saddle / back, socks, tail tip, ear tips, half face.
4. Small asymmetric details the owner will look for: "spot" ops (near an eye, on the head top, on a side) and
   eye patches. Small dots and brows (under ~14 mm on the toy) with "method": "satin".
5. Eyes colour if not dark brown/black, nose colour if not black, tail kind, proportions only when they are a
   breed trait (a corgi's short legs: shape legs length 0.7; a dachshund's long body).
6. Owner notes (if any) win over what you see.

Sides: L / R are the PET's own left / right. When the pet faces the camera, its left eye is on the RIGHT of the
image. Say the side in "seen" for every asymmetric marking and double-check it against the image.
Colours: pick the yarn colour of the lit coat, not shadows or highlights, as "#rrggbb"; at most 6 colours.
Use only the ops in the reference. Answer in the JSON format: seen, explanation, ops_json."""


def system_prompt():
    return INSTRUCTIONS + "\n\n<customisation_ops>\n" + (custom.__doc__ or "") + "\n</customisation_ops>"


def _image_block(path):
    data = Path(path).read_bytes()
    mt = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    if mt not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        raise ValueError(f"unsupported image type {mt}")
    return {"type": "image", "source": {"type": "base64", "media_type": mt,
                                        "data": base64.standard_b64encode(data).decode()}}


def build_messages(spec, image_path, notes=""):
    system = [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}]
    text = ("<base_toy>\n" + edit.summary(spec) + "\n" + custom.describe(spec) + "\n</base_toy>\n\n"
            + (f"<owner_notes>\n{notes.strip()}\n</owner_notes>\n\n" if notes and notes.strip() else "")
            + "Describe this pet as customisation ops.")
    return system, [{"role": "user", "content": [_image_block(image_path), {"type": "text", "text": text}]}]


def _ask(anthropic, client, system, messages):
    try:
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=system, messages=messages,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    except anthropic.AuthenticationError as ex:
        raise LLMUnavailable("the Claude API rejected the key (check ANTHROPIC_API_KEY)") from ex
    except anthropic.APIStatusError as ex:
        raise LLMError(f"Claude API error {ex.status_code}: {ex.message}") from ex
    except anthropic.APIConnectionError as ex:
        raise LLMError("cannot reach the Claude API (network)") from ex
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise LLMError(f"the model stopped: {resp.stop_reason}")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise LLMError("empty answer from the model")
    data = json.loads(text)
    try:
        ops = json.loads(data["ops_json"]) if data["ops_json"].strip() else []
    except json.JSONDecodeError as ex:
        ops = ex
    return resp, data, ops


def photo_to_ops(spec, image_path, notes="", client=None):
    """Photo -> {"ops", "seen", "explanation", "spec", "report", "attempts"} (raises LLMUnavailable / LLMError)."""
    anthropic, cl = _client() if client is None else (__import__("anthropic"), client)
    system, messages = build_messages(spec, image_path, notes)
    last = None
    for attempt in range(2):
        resp, data, ops = _ask(anthropic, cl, system, messages)
        try:
            if isinstance(ops, Exception) or not isinstance(ops, list):
                raise edit.EditError(f"ops_json must be a JSON array: {ops}")
            new, report = edit.apply(spec, ops)
            return {"ops": ops, "seen": data.get("seen", ""), "explanation": data.get("explanation", ""),
                    "spec": new, "report": report, "attempts": attempt + 1}
        except edit.EditError as ex:
            last = str(ex)
            messages = messages + [{"role": "assistant", "content": resp.content},
                                   {"role": "user", "content": f"Applying these ops failed: {last}\n"
                                                               "Return a corrected answer (same JSON format)."}]
    raise LLMError(f"the model's ops did not apply: {last}")


def load_base(base):
    p = BASES.get(base, Path(base))
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="pet photo -> customised amikit toy")
    ap.add_argument("photo", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--base", default="dog", help="dog | cat | path to a spec")
    ap.add_argument("--notes", default="", help="owner's notes (they win over the photo)")
    ap.add_argument("--ops", help="apply these ops (JSON file) instead of asking the model")
    ap.add_argument("--name", help="toy id (default: pet_<photo name>)")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_intermixed_args(argv)
    spec = load_base(a.base)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.ops:
        ops = json.loads(Path(a.ops).read_text(encoding="utf-8"))
        new, report = edit.apply(spec, ops)
        res = {"ops": ops, "seen": "", "explanation": "ops given", "spec": new, "report": report}
    else:
        if not a.photo:
            ap.error("a photo (or --ops) is required")
        try:
            res = photo_to_ops(spec, a.photo, a.notes)
        except (LLMUnavailable, LLMError) as ex:
            print(f"error: {ex}", file=sys.stderr); return 1
    new = res["spec"]
    new["name"] = a.name or ("pet_" + Path(a.photo or a.ops).stem)
    (out / "ops.json").write_text(json.dumps(res["ops"], ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "seen.txt").write_text(res["seen"] + "\n\n" + res["explanation"] + "\n", encoding="utf-8")
    sp = out / f"{new['name']}.json"
    sp.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
    from .build import build
    build(str(sp), str(out), a.photo, a.fast)
    print(res["explanation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
