"""Operator-configured experimental evaluators (``AVD_EXPERIMENTAL_EVALUATORS``).

There is deliberately no automatic plugin discovery (no entry points, no plugin directory):
code runs only when the operator names it, as ``"package.module:ClassName"``. Rules that keep the
official protocol intact:

* **Lowest precedence.** Experimental evaluators are placed *before* the official ones, and later
  evaluators win per dimension, so a plugin can only fill dimensions the official evaluators
  abstain on. It can never replace an official score.
* **Recorded.** Their names and versions enter the pipeline config (hence the config hash, so
  affected evaluations are re-run and never mixed with official ones) and every evaluation's
  ``versions["experimental_evaluators"]``, which exports carry.
* **No credentials passed.** An evaluator receives only the ``EvaluationInput`` and the genre
  profile. A Python plugin runs in-process, though, and could read the environment itself:
  only configure code you have reviewed and trust, as with any dependency.
"""

from __future__ import annotations

import importlib
from typing import Any

REQUIRED = ("name", "version", "dimensions", "requires", "synthetic", "evaluate")


class PluginError(RuntimeError):
    pass


def load_evaluators(specs: list[str]) -> list[Any]:
    out = []
    for spec in specs:
        module, _, cls = spec.partition(":")
        if not module or not cls:
            raise PluginError(f"{spec!r}: use 'package.module:ClassName'")
        try:
            obj = getattr(importlib.import_module(module), cls)()
        except Exception as exc:
            raise PluginError(f"{spec!r} could not be loaded: {type(exc).__name__}: {exc}") from exc
        missing = [a for a in REQUIRED if not hasattr(obj, a)]
        if missing:
            raise PluginError(f"{spec!r} is not an Evaluator (missing {', '.join(missing)})")
        out.append(obj)
    return out
