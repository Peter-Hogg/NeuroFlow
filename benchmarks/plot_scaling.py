"""Render a dependency-free SVG from a synthetic scaling result."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from neuroflow.benchmarking import validate_benchmark_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    suite = json.loads(args.input.read_text())
    records = suite.get("records")
    if not isinstance(records, list) or not records:
        parser.error("input has no scaling records")
    points: list[tuple[float, float, float]] = []
    for record in records:
        validate_benchmark_record(record)
        selected = float(record["source"]["selected_bytes"])
        rss = float(record["execution"]["peak_rss_bytes"])
        seconds = float(record["execution"]["wall_time_seconds"])
        points.append((selected, rss, seconds))
    svg = _svg(points, title=str(suite.get("suite_name", "scaling")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)


def _svg(points: list[tuple[float, float, float]], *, title: str) -> str:
    width, height = 900, 520
    left, top, plot_width, plot_height = 90, 70, 740, 360
    max_x = max(item[0] for item in points) or 1
    max_y = max(item[1] for item in points) or 1

    def x(value: float) -> float:
        return left + value / max_x * plot_width

    def y(value: float) -> float:
        return top + plot_height - value / max_y * plot_height

    coordinates = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b, _ in points)
    circles = "\n".join(
        f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="5"><title>'
        f"selected={a:.0f} B, peak RSS={b:.0f} B, wall={seconds:.6f} s"
        "</title></circle>"
        for a, b, seconds in points
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>text{{font:14px sans-serif}} .axis{{stroke:#333}} polyline{{fill:none;stroke:#276fbf;stroke-width:3}} circle{{fill:#d1495b}}</style>
<text x="{width / 2}" y="30" text-anchor="middle" font-size="20">{html.escape(title)}</text>
<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>
<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>
<polyline points="{coordinates}"/>{circles}
<text x="{left + plot_width / 2}" y="{height - 30}" text-anchor="middle">Selected logical bytes</text>
<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle">Peak RSS bytes</text>
</svg>\n"""


if __name__ == "__main__":
    main()
