"""Natural-language edit of a toy spec through Claude: text request (Russian or English) -> edit ops.

    from amikit.llm_edit import text_to_ops
    res = text_to_ops(spec, "уши побольше и повыше")   # {"ops": [...], "explanation": "..."}

    python -m amikit.llm_edit <spec.json> "<request>" <out.json>

The model sees SPEC.md, the edit-ops reference (amikit/edit.py docstring), edit.summary(spec) and the full spec,
and answers with structured JSON. The ops are validated with edit.apply (which builds the toy); on an EditError the
error goes back to the model once for a corrected answer.

Needs the `anthropic` package and ANTHROPIC_API_KEY in the environment (never stored or logged here).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

from . import edit

MODEL = "claude-opus-5-5"
HERE = Path(__file__).resolve().parent


class LLMUnavailable(RuntimeError):
    """No API key / SDK: the caller should answer 503."""


class LLMError(RuntimeError):
    """The model did not produce valid ops."""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string",
                        "description": "One or two short sentences, in the user's language, saying what the ops do."},
        "ops_json": {"type": "string",
                     "description": "A JSON array of edit operations, exactly as in the ops reference."},
    },
    "required": ["explanation", "ops_json"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You edit amigurumi toy specs for a crochet toy engine. A user describes a change in words
(Russian or English); you translate it into a list of edit operations on the spec.

How to work:
- Read the spec format (SPEC.md) and the edit-ops reference below; address parts, assembly entries and face items
  by their ids/names exactly as they appear in the given spec and summary.
- For a pet's look (colours, spots, markings, ear / tail kinds, longer / shorter parts) prefer the customisation
  ops (colour, spot, marking, ears, tail, shape, eyes, nose): they place things by the toy's own landmarks.
  "Left" / "right" are the toy's own (its left eye is on the viewer's right).
- Prefer the smallest set of ops that does what was asked. Use "set" on dotted paths for numbers, vectors and
  rounds; "scale_part" to make a piece bigger/smaller; "recolour" to change a named colour everywhere; "add" and
  "remove" for new or deleted parts/entries/face items.
- Shapes only change through rounds (stitch counts), never by inventing fields. Keep stitch arithmetic valid:
  max +100% / -50% per round. When moving things sewn on a piece, stay within that piece's rounds.
- Mirrored entries describe the toy's left side (+X, positive az); the right copy follows automatically.
- If the request is ambiguous, choose the most likely reading and say so in the explanation. If it cannot be
  expressed with ops, return an empty list and explain why.
- explanation: in the user's language, short. ops_json: a JSON array (as a string) of op objects, nothing else.
"""


def _edit_reference():
    from . import custom
    return (edit.__doc__ or "") + "\n\nCustomisation ops in full (amikit/custom.py):\n" + (custom.__doc__ or "")


def system_prompt():
    spec_md = (HERE / "SPEC.md").read_text(encoding="utf8")
    return (INSTRUCTIONS + "\n\n<spec_format>\n" + spec_md + "\n</spec_format>\n\n<edit_ops_reference>\n"
            + _edit_reference() + "\n</edit_ops_reference>")


def user_prompt(spec, text):
    from . import custom
    return ("<summary>\n" + edit.summary(spec) + "\n" + custom.describe(spec) + "\n</summary>\n\n<spec_json>\n"
            + json.dumps(spec, ensure_ascii=False, indent=1) + "\n</spec_json>\n\n<request>\n" + text.strip()
            + "\n</request>")


def build_messages(spec, text):
    """The request pieces (exposed for tests): system blocks + first user message."""
    system = [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}]
    return system, [{"role": "user", "content": user_prompt(spec, text)}]


def _client():
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise LLMUnavailable("set ANTHROPIC_API_KEY to use text commands")
    try:
        import anthropic
    except ImportError as ex:
        raise LLMUnavailable("the 'anthropic' package is not installed (pip install anthropic)") from ex
    return anthropic, anthropic.Anthropic()


def _ask(anthropic, client, system, messages):
    try:
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system,
            messages=messages,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as ex:
        raise LLMUnavailable("the Claude API rejected the key (check ANTHROPIC_API_KEY)") from ex
    except anthropic.RateLimitError as ex:
        raise LLMError("Claude API rate limit, try again in a minute") from ex
    except anthropic.APIStatusError as ex:
        raise LLMError(f"Claude API error {ex.status_code}: {ex.message}") from ex
    except anthropic.APIConnectionError as ex:
        raise LLMError("cannot reach the Claude API (network)") from ex
    if resp.stop_reason == "refusal":
        raise LLMError("the model declined this request")
    if resp.stop_reason == "max_tokens":
        raise LLMError("the model's answer was cut off")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise LLMError("empty answer from the model")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as ex:
        raise LLMError(f"the model's answer is not JSON: {ex}") from ex
    try:
        ops = json.loads(data["ops_json"]) if data["ops_json"].strip() else []
    except json.JSONDecodeError as ex:
        ops = ex  # reported back to the model as an error
    return resp, text, data.get("explanation", ""), ops


def text_to_ops(spec, text, client=None):
    """Request in words -> {"ops": [...], "explanation": str, "spec": new_spec, "report": {...}}.

    Raises LLMUnavailable (no key/SDK) or LLMError (no valid answer after one correction round)."""
    anthropic, cl = _client() if client is None else (__import__("anthropic"), client)
    system, messages = build_messages(spec, text)
    last_err = None
    for attempt in range(2):
        resp, raw, explanation, ops = _ask(anthropic, cl, system, messages)
        try:
            if isinstance(ops, Exception):
                raise edit.EditError(f"ops_json is not valid JSON: {ops}")
            if not isinstance(ops, list) or not all(isinstance(o, dict) for o in ops):
                raise edit.EditError("ops_json must be a JSON array of op objects")
            new_spec, report = edit.apply(spec, ops) if ops else (spec, {"log": [], "flags": []})
            return {"ops": ops, "explanation": explanation, "spec": new_spec, "report": report,
                    "attempts": attempt + 1}
        except edit.EditError as ex:
            last_err = str(ex)
            messages = messages + [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": f"Applying these ops failed: {last_err}\n"
                                            "Return a corrected answer (same JSON format)."},
            ]
    raise LLMError(f"the model's ops did not apply: {last_err}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    for s in (sys.stdout, sys.stderr):  # Russian text on Windows consoles
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    if len(argv) != 3:
        print(__doc__.split("\n\n")[1]); return 2
    spec_path, request, out = argv
    spec = json.loads(Path(spec_path).read_text(encoding="utf8"))
    try:
        res = text_to_ops(spec, request)
    except (LLMUnavailable, LLMError) as ex:
        print(f"error: {ex}", file=sys.stderr); return 1
    Path(out).write_text(json.dumps(res["spec"], ensure_ascii=False, indent=2), encoding="utf8")
    print(res["explanation"])
    print(json.dumps(res["ops"], ensure_ascii=False, indent=1))
    for f in res["report"].get("flags", []):
        print("flag:", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
