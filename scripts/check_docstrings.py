"""Check that every public callable documents every input: name, type and purpose.

For each public function, method and class (whose constructor parameters belong
in the class docstring, numpydoc style) in ``ume``, ``ume.testing`` and
``ume._core``, compare the signature's parameters with the docstring's
``Parameters`` section. Reports parameters that are missing, have no type
(``name : type``), or have no description. Exits with status 1 if anything is
missing, so it can run in CI.

Run: ``uv run python scripts/check_docstrings.py``
"""

from __future__ import annotations

import inspect
import re
import sys
from collections.abc import Callable, Iterator
from typing import Any

import ume
import ume.testing
from ume import _core

SKIP_PARAMS = {"self", "cls", "args", "kwargs"}


def documented(doc: str) -> dict[str, tuple[bool, bool]]:
    """``{name: (has_type, has_description)}`` from a numpydoc Parameters section."""
    lines = inspect.cleandoc(doc).splitlines()
    out: dict[str, tuple[bool, bool]] = {}
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Parameters") + 2
    except StopIteration:
        return out
    i = start
    while i < len(lines):
        line = lines[i]
        if line and not line.startswith(" ") and i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            break  # the next section
        if line and not line.startswith(" "):
            names, _, typ = line.partition(":")
            desc = []
            j = i + 1
            while j < len(lines) and (lines[j].startswith(" ") or not lines[j].strip()):
                if lines[j].strip():
                    desc.append(lines[j])
                j += 1
            for name in re.split(r",\s*", names.strip()):
                name = name.strip().lstrip("*")
                if name:
                    out[name] = (bool(typ.strip()) or "," in names, bool(desc))
            i = j
            continue
        i += 1
    return out


def params(obj: Callable[..., Any]) -> list[str]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    return [
        p.name
        for p in sig.parameters.values()
        if p.name not in SKIP_PARAMS and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def public_objects() -> Iterator[tuple[str, Any]]:
    for module, names in ((ume, ume.__all__), (ume.testing, ume.testing.__all__)):
        for name in names:
            obj = getattr(module, name)
            if callable(obj):
                yield f"{module.__name__}.{name}", obj
    for name in dir(_core):
        if not name.startswith("_"):
            yield f"ume._core.{name}", getattr(_core, name)


def check(qualname: str, obj: Any, doc_owner: Any, problems: list[str]) -> None:
    names = params(obj)
    if not names:
        return
    doc = inspect.getdoc(doc_owner) or ""
    found = documented(doc)
    for name in names:
        if name not in found:
            problems.append(f"{qualname}: parameter `{name}` is not documented")
        else:
            has_type, has_desc = found[name]
            if not has_type:
                problems.append(f"{qualname}: parameter `{name}` has no type")
            if not has_desc:
                problems.append(f"{qualname}: parameter `{name}` has no description")


def main() -> int:
    problems: list[str] = []
    for qualname, obj in public_objects():
        if inspect.isclass(obj):
            check(f"{qualname}()", obj, obj, problems)  # constructor parameters, in the class docstring
            for attr, member in vars(obj).items():
                if attr.startswith("_") and attr not in ("__add__",):
                    continue
                if inspect.isfunction(member) or inspect.ismethoddescriptor(member) or callable(member):
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if callable(member) and not inspect.isclass(member):
                        check(f"{qualname}.{attr}", member, member, problems)
        else:
            check(qualname, obj, obj, problems)
    for p in problems:
        print(p)
    print(f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
