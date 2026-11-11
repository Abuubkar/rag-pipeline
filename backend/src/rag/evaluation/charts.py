"""Before/after charts as plain SVG (no plotting library), one consistent style for every chart.

Form: small multiples. One panel per metric, one horizontal bar per experiment, all on a shared
0–1 scale. The baseline ("before") is gray and every later experiment is blue, so the eye goes to
what changed. A hairline in each panel marks the baseline value. Every bar carries its value at
the tip (the gray is deliberately low-contrast, so the number, the axis label and the table in
docs/results/README.md carry the meaning, never color alone).

Dark mode: the SVG has its own light/dark colors via prefers-color-scheme, so it reads well on
GitHub in either theme. Hovering a bar (when the SVG is opened directly) shows a <title> tooltip.
"""

from html import escape

# Colors: validated with the dataviz palette checker (blue vs gray: CVD ΔE 22, normal ΔE 23+).
_STYLE = """
  .surface { fill: #fcfcfb; }
  .title { fill: #0b0b0b; font-weight: 600; font-size: 14px; }
  .subtitle, .panel-title { fill: #52514e; font-size: 12px; }
  .panel-title { font-weight: 600; }
  .label { fill: #0b0b0b; font-size: 12px; }
  .value, .tick { fill: #52514e; font-size: 11px; font-variant-numeric: tabular-nums; }
  .value { paint-order: stroke; stroke: #fcfcfb; stroke-width: 4px; stroke-linejoin: round; }
  .grid { stroke: #e1e0d9; stroke-width: 1; }
  .axis { stroke: #c3c2b7; stroke-width: 1; }
  .ref { stroke: #898781; stroke-width: 1; }
  .bar-baseline { fill: #b5b3ac; }
  .bar-run { fill: #2a78d6; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .title, .label { fill: #f0efec; }
    .subtitle, .panel-title, .value, .tick { fill: #c3c2b7; }
    .value { stroke: #1a1a19; }
    .grid { stroke: #2c2c2a; }
    .axis { stroke: #383835; }
    .bar-baseline { fill: #74726c; }
    .bar-run { fill: #3987e5; }
  }
"""

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
_ROW, _BAR = 30, 18          # row pitch and bar thickness (≤ 24px)
_LABEL_W, _PANEL_W, _GAP = 210, 150, 22
_VALUE_ROOM = 34             # space right of the longest bar for its value
_TOP, _BOTTOM, _PAD = 74, 34, 16


def _bar_path(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Square at the baseline (left), 4px rounded data end (right)."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z")


_NAMES = {"fact_recall": "fact recall", "mrr": "MRR", "all": "all types"}


def small_multiples(
    title: str, subtitle: str, labels: list[str], metrics: list[str],
    values: list[dict[str, float | None]],
) -> str:
    """labels[i] is one experiment (labels[0] = baseline); values[i][metric] is its score (0–1)."""
    panel_w = _PANEL_W if len(metrics) <= 4 else 125  # keep the chart within GitHub's ~1,000px column
    plot_w = panel_w - _VALUE_ROOM
    width = _PAD + _LABEL_W + len(metrics) * (panel_w + _GAP) - _GAP + _PAD
    height = _TOP + len(labels) * _ROW + _BOTTOM
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" font-family=\'{_FONT}\' role="img" aria-label="{escape(title)}">',
        f"<style>{_STYLE}</style>",
        f'<rect class="surface" width="{width}" height="{height}" rx="8"/>',
        f'<text class="title" x="{_PAD}" y="26">{escape(title)}</text>',
        f'<text class="subtitle" x="{_PAD}" y="44">{escape(subtitle)}</text>',
    ]
    for i, label in enumerate(labels):
        y = _TOP + i * _ROW + _ROW / 2 + 4
        out.append(f'<text class="label" x="{_PAD}" y="{y:.1f}">{escape(label)}</text>')

    for m, metric in enumerate(metrics):
        x0 = _PAD + _LABEL_W + m * (panel_w + _GAP)
        bottom = _TOP + len(labels) * _ROW
        out.append(f'<text class="panel-title" x="{x0}" y="{_TOP - 12}">{escape(_NAMES.get(metric, metric))}</text>')
        for tick in (0, 0.5, 1):
            tx = x0 + tick * plot_w
            out.append(f'<line class="{"axis" if tick == 0 else "grid"}" x1="{tx:.1f}" y1="{_TOP - 4}" '
                       f'x2="{tx:.1f}" y2="{bottom}"/>')
            out.append(f'<text class="tick" x="{tx:.1f}" y="{bottom + 16}" text-anchor="middle">{tick:g}</text>')
        base = values[0].get(metric)
        if base is not None:  # drawn first, so bars and value labels sit on top of it
            bx = x0 + base * plot_w
            out.append(f'<line class="ref" x1="{bx:.1f}" y1="{_TOP - 4}" x2="{bx:.1f}" y2="{bottom}"/>')
        for i, (label, row) in enumerate(zip(labels, values)):
            v = row.get(metric)
            if v is None:
                continue
            y = _TOP + i * _ROW + (_ROW - _BAR) / 2
            w = max(v, 0) * plot_w
            cls = "bar-baseline" if i == 0 else "bar-run"
            delta = "" if i == 0 or base is None else f" ({v - base:+.2f} vs baseline)"
            out.append(f'<path class="{cls}" d="{_bar_path(x0, y, w, _BAR)}">'
                       f"<title>{escape(label)} · {_NAMES.get(metric, metric)} {v:.2f}{delta}</title></path>")
            out.append(f'<text class="value" x="{x0 + w + 5:.1f}" y="{y + _BAR / 2 + 4:.1f}">{v:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)
