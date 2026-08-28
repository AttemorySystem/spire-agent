"""Render one Winning Path decision record as compact plain text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def render(payload: Mapping[str, Any]) -> str:
    review = _mapping(payload.get("review"))
    analysis = _mapping(review.get("winning_path"))
    state = _mapping(analysis.get("state"))
    run = _mapping(state.get("run"))
    resolution = _mapping(analysis.get("resolution"))
    target = _mapping(analysis.get("target_plan"))
    lines = [
        f"command: {payload.get('command')}",
        f"source: {payload.get('source')}",
        f"confirmed: {payload.get('confirmed')}",
        f"policy: {review.get('policy')}",
        f"reason: {payload.get('reason')}",
        f"state: Act {run.get('act')} Floor {run.get('floor')}",
        "targets: " + ", ".join(map(str, _sequence(target.get("targets")))),
        "frontier: "
        + ", ".join(map(str, _sequence(resolution.get("frontier_choice_ids")))),
        "candidates:",
    ]
    for raw in _sequence(review.get("candidates")):
        row = _mapping(raw)
        template = _mapping(row.get("template"))
        transition = _mapping(row.get("transition"))
        expert = _mapping(row.get("expert"))
        detail = (
            f"template={template.get('level')} "
            f"transition={transition.get('level')} "
            f"expert={expert.get('level')}"
        )
        if template.get("route_id"):
            detail += f" route={template['route_id']}"
        lines.append(
            f"  [{row.get('choice_id')}] {row.get('name')}: "
            f"{'REJECTED ' if row.get('rejected') else ''}{detail}"
        )
    ranking = _mapping(resolution.get("card_preference"))
    if ranking:
        lines.append(
            "preference scores: "
            + json.dumps(ranking.get("scores"), ensure_ascii=False)
        )
        for raw in _sequence(ranking.get("comparisons")):
            row = _mapping(raw)
            lines.append(
                f"  {row.get('left')} vs {row.get('right')}: "
                f"{row.get('left_wins')}:{row.get('right_wins')} "
                f"bucket={row.get('bucket')} z={row.get('z')}"
            )
    if payload.get("llm_proposal") is not None:
        lines.append(
            "llm proposal: "
            + json.dumps(payload["llm_proposal"], ensure_ascii=False)
        )
    return "\n".join(lines)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[Any, ...]:
    return (
        tuple(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit("Winning Path log must contain a JSON object")
    print(render(payload))


if __name__ == "__main__":
    main()


__all__ = ["main", "render"]
