"""
The editor's edits to a finished document, checked and applied.

The CV editor shows the CV as ``latex_renderer.cv_view`` prints it. Every
printed segment names the CV-data field it came from (its ``src`` path, e.g.
``experience.0.company``), and every bullet list names its own path
(``bullets_src``). An edit may only address those paths:

- ``set`` a field the view prints (a comma-separated list field takes the
  items as a list or as comma-separated text),
- ``insert`` a bullet into a bullet list the view prints,
- ``delete`` a bullet from one.

Anything else is refused with a reason, so the editor can't write fields the
PDF doesn't print or change the CV's structure. Edits apply in order, each
checked against the view of the CV as the edits before it left it.

The cover letter's editable fields are its paragraphs, the recipient (the
salutation follows from it), the sign-off and, for a German letter, the
Betreff line.
"""

from __future__ import annotations

import copy
from typing import Any

from app.services.latex_renderer import (
    LETTER_PARAGRAPH_FIELDS,
    cover_letter_framing,
    cv_view,
    default_sign_off,
)

MAX_VALUE_CHARS = 5000


class EditRefused(ValueError):
    """An edit the document can't take; the message says why."""


# --------------------------------------------------------------------------- #
# CV
# --------------------------------------------------------------------------- #

def _segments(view: dict[str, Any]) -> list[dict[str, Any]]:
    header = view["header"]
    segs = [*header.get("name_segs", []), *header.get("label_segs", []), *header.get("summary_segs", [])]
    segs += [s for part in header.get("contact_segs", []) for s in part]
    for section in view["sections"]:
        for entry in section["entries"]:
            for row in entry.get("rows", []):
                segs += [*row.get("left_segs", []), *row.get("right_segs", [])]
            segs += [s for bullet in entry.get("bullet_segs", []) for s in bullet]
            segs += [*entry.get("label_segs", []), *entry.get("text_segs", [])]
    return segs


def editable_paths(document: dict[str, Any]) -> tuple[dict[str, str | None], set[str]]:
    """({field path: kind} for every field the CV's view prints, {bullet list paths})."""
    view = cv_view(document["cv"], document.get("section_order") or None,
                   convention=document.get("convention") or "international")
    fields = {path: seg.get("kind") for seg in _segments(view) for path in seg["src"]}
    lists = {
        path
        for section in view["sections"] for entry in section["entries"]
        for path in entry.get("bullets_src", [])
    }
    return fields, lists


def _parts(path: str) -> list[str | int]:
    return [int(p) if p.isdigit() else p for p in path.split(".")]


def _parent(cv: dict[str, Any], path: str) -> tuple[Any, str | int]:
    """The container holding ``path``'s last part, and that part."""
    *head, last = _parts(path)
    node: Any = cv
    for part in head:
        try:
            node = node[part]
        except (KeyError, IndexError, TypeError):
            raise EditRefused(f"There's no field at {path}.") from None
    return node, last


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise EditRefused(f"The value for {path} has to be text.")
    if len(value) > MAX_VALUE_CHARS:
        raise EditRefused(f"The value for {path} is longer than {MAX_VALUE_CHARS} characters.")
    return value.strip()


def _items(value: Any, path: str) -> list[str]:
    raw = value if isinstance(value, list) else _text(value, path).split(",")
    return [_text(item, path) for item in raw if isinstance(item, str) and item.strip()]


def apply_cv_edits(document: dict[str, Any], edits: list[dict[str, Any]]) -> dict[str, Any]:
    """The document's CV with ``edits`` ({op, path, value?, index?}) applied, in
    order; raises EditRefused on the first edit the CV can't take."""
    document = {**document, "cv": copy.deepcopy(document["cv"])}
    cv = document["cv"]
    for edit in edits:
        op, path = edit.get("op"), str(edit.get("path") or "")
        fields, lists = editable_paths(document)
        if op == "set":
            if path not in fields:
                raise EditRefused(f"{path} isn't a field the CV prints.")
            container, key = _parent(cv, path)
            current = container[key] if isinstance(container, dict) and key in container else None
            if isinstance(current, list) or fields[path] == "list":
                container[key] = _items(edit.get("value"), path)
            else:
                container[key] = _text(edit.get("value"), path)
        elif op == "insert":
            if path not in lists:
                raise EditRefused(f"{path} isn't a bullet list the CV prints.")
            container, key = _parent(cv, path)
            bullets = container.setdefault(key, []) if isinstance(container, dict) else None
            if not isinstance(bullets, list):
                raise EditRefused(f"{path} isn't a bullet list the CV prints.")
            text = _text(edit.get("value"), path)
            if not text:
                raise EditRefused("A new bullet needs text.")
            index = edit.get("index")
            bullets.insert(len(bullets) if index is None else max(0, min(int(index), len(bullets))), text)
        elif op == "delete":
            parent_path, _, index = path.rpartition(".")
            if parent_path not in lists or not index.isdigit():
                raise EditRefused(f"{path} isn't a bullet the CV prints.")
            container, key = _parent(cv, parent_path)
            bullets = container.get(key) if isinstance(container, dict) else None
            if not isinstance(bullets, list) or int(index) >= len(bullets):
                raise EditRefused(f"{path} isn't a bullet the CV prints.")
            del bullets[int(index)]
        else:
            raise EditRefused(f"Unknown edit '{op}'.")
    return cv


# --------------------------------------------------------------------------- #
# Cover letter
# --------------------------------------------------------------------------- #

def _is_german(content: dict[str, Any]) -> bool:
    return (content.get("language") or "english").lower() == "german"


def letter_fields(document: dict[str, Any]) -> dict[str, str]:
    """The letter's editable fields as the PDF prints them."""
    content, recipient = document["content"], document.get("recipient") or ""
    fields = {field: str(content.get(field) or "") for field in LETTER_PARAGRAPH_FIELDS}
    fields["recipient"] = recipient
    fields["sign_off"] = cover_letter_framing(content, recipient)["sign_off"] if recipient else ""
    if _is_german(content):
        fields["betreff"] = str(content.get("betreff") or "")
    return fields


def apply_letter_edits(document: dict[str, Any], changes: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """(content, recipient) with ``changes`` ({field: text}) applied; raises
    EditRefused for a field the letter doesn't have or an empty recipient."""
    content = dict(document["content"])
    recipient = document.get("recipient") or ""
    allowed = set(letter_fields(document))
    for field, value in changes.items():
        if field not in allowed:
            raise EditRefused(f"{field} isn't a field of this letter.")
        text = _text(value, field)
        if field == "recipient":
            if not text:
                raise EditRefused("The letter needs a recipient (a name, or e.g. 'Hiring Team').")
            recipient = text
        elif field != "sign_off":
            content[field] = text
    if "sign_off" in changes:
        # Kept only when it differs from the sign-off the letter gets anyway,
        # so a later change of recipient still picks the fitting one.
        sign_off = _text(changes["sign_off"], "sign_off")
        key = "grussformel" if _is_german(content) else "sign_off"
        if sign_off and sign_off != default_sign_off(content, recipient):
            content[key] = sign_off
        else:
            content.pop(key, None)
    return content, recipient
