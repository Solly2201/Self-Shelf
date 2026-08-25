/* Minimal SVG chart helpers for the Self-Shelf dashboard.
   Rendering only — every plotted number comes from the backend engine. */

(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";

  const COLORS = {
    accent: "#7C5CFF",
    accentSoft: "rgba(124, 92, 255, 0.16)",
    line: "#8B8B94",
    grid: "#222227",
    text: "#6F6F78",
    positive: "#4CAF7D",
    warning: "#D6A84F",
    danger: "#D96767",
    neutral: "#3d3d45",
  };

  function el(tag, attrs, children) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      node.setAttribute(k, v);
    }
    for (const child of children || []) node.appendChild(child);
    return node;
  }

  function textEl(content, attrs) {
    const node = el("text", Object.assign({
      fill: COLORS.text,
      "font-size": "11",
      "font-family": "inherit",
    }, attrs));
    node.textContent = content;
    return node;
  }

  function niceTicks(min, max, count) {
    if (min === max) { max = min + 1; }
    const span = max - min;
    const step = Math.pow(10, Math.floor(Math.log10(span / count)));
    const err = (span / count) / step;
    let unit = step;
    if (err >= 7.5) unit = step * 10;
    else if (err >= 3.5) unit = step * 5;
    else if (err >= 1.5) unit = step * 2;
    const ticks = [];
    let t = Math.ceil(min / unit) * unit;
    while (t <= max + 1e-9) { ticks.push(t); t += unit; }
    return ticks;
  }

  let tipNode = null;
  function tip() {
    if (!tipNode) {
      tipNode = document.createElement("div");
      tipNode.className = "chart-tip";
      document.body.appendChild(tipNode);
    }
    return tipNode;
  }

  function showTip(html, x, y) {
    const node = tip();
    node.innerHTML = html;
    node.style.display = "block";
    const rect = node.getBoundingClientRect();
    const left = Math.min(x + 14, window.innerWidth - rect.width - 12);
    const top = Math.min(y + 14, window.innerHeight - rect.height - 12);
    node.style.left = left + "px";
    node.style.top = top + "px";
  }

  function hideTip() { if (tipNode) tipNode.style.display = "none"; }

  /* Line/curve chart with optional vertical markers and hover readout.
     opts: { width, height, series: [{points: [{x, y, meta}], color, area}],
             markers: [{x, label, color, dash}], xFormat, yFormat,
             tooltip(point) -> html, xLabel, yLabel } */
  function lineChart(container, opts) {
    const width = opts.width || 640;
    const height = opts.height || 280;
    const pad = { top: 16, right: 18, bottom: 34, left: 56 };

    const allPoints = opts.series.flatMap((s) => s.points);
    if (!allPoints.length) return;
    const xs = allPoints.map((p) => p.x);
    const ys = allPoints.map((p) => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    const yPadding = (yMax - yMin || Math.abs(yMax) || 1) * 0.08;
    yMin -= yPadding; yMax += yPadding;

    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const sx = (x) => pad.left + ((x - xMin) / (xMax - xMin || 1)) * plotW;
    const sy = (y) => pad.top + plotH - ((y - yMin) / (yMax - yMin || 1)) * plotH;

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": opts.ariaLabel || "chart",
    });

    for (const t of niceTicks(yMin, yMax, 4)) {
      svg.appendChild(el("line", {
        x1: pad.left, x2: width - pad.right, y1: sy(t), y2: sy(t),
        stroke: COLORS.grid, "stroke-width": 1,
      }));
      svg.appendChild(textEl(
        (opts.yFormat || String)(t),
        { x: pad.left - 8, y: sy(t) + 3.5, "text-anchor": "end" }
      ));
    }
    for (const t of niceTicks(xMin, xMax, 6)) {
      svg.appendChild(textEl(
        (opts.xFormat || String)(t),
        { x: sx(t), y: height - pad.bottom + 18, "text-anchor": "middle" }
      ));
    }

    if (typeof opts.zeroLine === "number"
        && opts.zeroLine >= yMin && opts.zeroLine <= yMax) {
      svg.appendChild(el("line", {
        x1: pad.left, x2: width - pad.right,
        y1: sy(opts.zeroLine), y2: sy(opts.zeroLine),
        stroke: COLORS.text, "stroke-width": 1, "stroke-dasharray": "2 4",
        opacity: 0.5,
      }));
    }

    for (const s of opts.series) {
      const pts = s.points;
      const path = pts.map(
        (p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`
      ).join("");
      if (s.area) {
        const areaPath = path
          + `L${sx(pts[pts.length - 1].x).toFixed(1)},${(pad.top + plotH).toFixed(1)}`
          + `L${sx(pts[0].x).toFixed(1)},${(pad.top + plotH).toFixed(1)}Z`;
        svg.appendChild(el("path", {
          d: areaPath, fill: s.areaColor || COLORS.accentSoft, stroke: "none",
        }));
      }
      svg.appendChild(el("path", {
        d: path, fill: "none",
        stroke: s.color || COLORS.accent,
        "stroke-width": s.width || 2,
        "stroke-linejoin": "round", "stroke-linecap": "round",
        "stroke-dasharray": s.dash || "none",
      }));
    }

    for (const m of opts.markers || []) {
      if (m.x < xMin - 1e-9 || m.x > xMax + 1e-9) continue;
      svg.appendChild(el("line", {
        x1: sx(m.x), x2: sx(m.x), y1: pad.top, y2: pad.top + plotH,
        stroke: m.color || COLORS.text,
        "stroke-width": 1,
        "stroke-dasharray": m.dash || "3 4",
        opacity: m.opacity != null ? m.opacity : 0.8,
      }));
      if (m.label) {
        svg.appendChild(textEl(m.label, {
          x: sx(m.x), y: pad.top - 4, "text-anchor": "middle",
          fill: m.color || COLORS.text, "font-size": "10.5",
          "font-weight": "500",
        }));
      }
      if (m.y != null) {
        svg.appendChild(el("circle", {
          cx: sx(m.x), cy: sy(m.y), r: 4,
          fill: m.color || COLORS.accent, stroke: "#0B0B0D",
          "stroke-width": 1.5,
        }));
      }
    }

    if (opts.tooltip) {
      const hover = el("rect", {
        x: pad.left, y: pad.top, width: plotW, height: plotH,
        fill: "transparent",
      });
      const cursor = el("line", {
        x1: 0, x2: 0, y1: pad.top, y2: pad.top + plotH,
        stroke: COLORS.text, "stroke-width": 1, opacity: 0, "pointer-events": "none",
      });
      svg.appendChild(cursor);
      const primary = opts.series[0].points;
      hover.addEventListener("mousemove", (evt) => {
        const rect = svg.getBoundingClientRect();
        const px = ((evt.clientX - rect.left) / rect.width) * width;
        const dataX = xMin + ((px - pad.left) / plotW) * (xMax - xMin);
        let best = primary[0];
        for (const p of primary) {
          if (Math.abs(p.x - dataX) < Math.abs(best.x - dataX)) best = p;
        }
        cursor.setAttribute("x1", sx(best.x));
        cursor.setAttribute("x2", sx(best.x));
        cursor.setAttribute("opacity", 0.35);
        showTip(opts.tooltip(best), evt.clientX, evt.clientY);
      });
      hover.addEventListener("mouseleave", () => {
        cursor.setAttribute("opacity", 0);
        hideTip();
      });
      svg.appendChild(hover);
    }

    container.classList.add("chart-box");
    container.innerHTML = "";
    container.appendChild(svg);
  }

  /* Vertical bar chart. opts: { items: [{label, value, color, title}],
     yFormat, height } */
  function barChart(container, opts) {
    const width = opts.width || 520;
    const height = opts.height || 220;
    const pad = { top: 14, right: 10, bottom: 32, left: 44 };
    const items = opts.items;
    if (!items.length) return;
    const maxV = Math.max(...items.map((d) => d.value), 1);
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const slot = plotW / items.length;
    const barW = Math.min(slot * 0.55, 48);

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img", "aria-label": opts.ariaLabel || "bar chart",
    });

    for (const t of niceTicks(0, maxV, 3)) {
      const y = pad.top + plotH - (t / maxV) * plotH;
      svg.appendChild(el("line", {
        x1: pad.left, x2: width - pad.right, y1: y, y2: y,
        stroke: COLORS.grid, "stroke-width": 1,
      }));
      svg.appendChild(textEl((opts.yFormat || String)(t), {
        x: pad.left - 8, y: y + 3.5, "text-anchor": "end",
      }));
    }

    items.forEach((d, i) => {
      const h = (d.value / maxV) * plotH;
      const x = pad.left + slot * i + (slot - barW) / 2;
      const bar = el("rect", {
        x, y: pad.top + plotH - h, width: barW, height: Math.max(h, d.value > 0 ? 2 : 0),
        rx: 3, fill: d.color || COLORS.accent, opacity: 0.9,
      });
      const label = d.title || `${d.label}: ${d.value}`;
      bar.appendChild(el("title", {}, [document.createTextNode(label)]));
      svg.appendChild(bar);
      svg.appendChild(textEl(d.label, {
        x: pad.left + slot * i + slot / 2, y: height - pad.bottom + 18,
        "text-anchor": "middle",
      }));
    });

    container.classList.add("chart-box");
    container.innerHTML = "";
    container.appendChild(svg);
  }

  /* Grouped horizontal bars: rows of [label, series values].
     opts: { rows: [{label, values: [{value, color, name}]}], format } */
  function groupedBars(container, opts) {
    const width = opts.width || 520;
    const rowH = 46;
    const pad = { top: 6, right: 60, bottom: 6, left: 110 };
    const rows = opts.rows;
    if (!rows.length) return;
    const height = pad.top + pad.bottom + rows.length * rowH;
    const maxV = Math.max(
      ...rows.flatMap((r) => r.values.map((v) => v.value)), 1
    );
    const plotW = width - pad.left - pad.right;

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img", "aria-label": opts.ariaLabel || "grouped bars",
    });

    rows.forEach((row, i) => {
      const y0 = pad.top + i * rowH;
      svg.appendChild(textEl(row.label, {
        x: pad.left - 10, y: y0 + rowH / 2 + 4, "text-anchor": "end",
        fill: "#A1A1A8", "font-size": "12",
      }));
      const barH = 9;
      const gap = 5;
      const totalH = row.values.length * barH + (row.values.length - 1) * gap;
      row.values.forEach((v, j) => {
        const w = (v.value / maxV) * plotW;
        const y = y0 + (rowH - totalH) / 2 + j * (barH + gap);
        const bar = el("rect", {
          x: pad.left, y, width: Math.max(w, v.value > 0 ? 2 : 0), height: barH,
          rx: 2.5, fill: v.color, opacity: 0.9,
        });
        bar.appendChild(el("title", {}, [document.createTextNode(
          `${row.label} — ${v.name}: ${(opts.format || String)(v.value)}`
        )]));
        svg.appendChild(bar);
        svg.appendChild(textEl((opts.format || String)(v.value), {
          x: pad.left + Math.max(w, 2) + 6, y: y + barH - 1.5,
          "font-size": "10.5",
        }));
      });
    });

    container.classList.add("chart-box");
    container.innerHTML = "";
    container.appendChild(svg);
  }

  /* Dumbbell rows: current -> recommended price per product. */
  function dumbbells(container, opts) {
    const width = opts.width || 860;
    const rowH = 34;
    const pad = { top: 24, right: 60, bottom: 8, left: 230 };
    const rows = opts.rows;
    if (!rows.length) return;
    const height = pad.top + pad.bottom + rows.length * rowH;
    const values = rows.flatMap((r) => [r.from, r.to]);
    const vMin = Math.min(...values), vMax = Math.max(...values);
    const span = vMax - vMin || 1;
    const plotW = width - pad.left - pad.right;
    const sx = (v) => pad.left + ((v - vMin) / span) * plotW;

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img", "aria-label": opts.ariaLabel || "price changes",
    });

    svg.appendChild(textEl("current", {
      x: pad.left, y: 12, "font-size": "10.5",
    }));
    svg.appendChild(textEl("recommended", {
      x: pad.left + 74, y: 12, fill: COLORS.accent, "font-size": "10.5",
    }));

    rows.forEach((row, i) => {
      const y = pad.top + i * rowH + rowH / 2;
      const shortLabel = row.label.length > 30
        ? row.label.slice(0, 29).trimEnd() + "…" : row.label;
      const name = textEl(shortLabel, {
        x: pad.left - 10, y: y + 4, "text-anchor": "end",
        fill: "#A1A1A8", "font-size": "12",
      });
      name.appendChild(el("title", {}, [document.createTextNode(row.label)]));
      svg.appendChild(name);
      const x1 = sx(row.from), x2 = sx(row.to);
      if (Math.abs(x2 - x1) > 0.5) {
        svg.appendChild(el("line", {
          x1, x2, y1: y, y2: y, stroke: COLORS.neutral, "stroke-width": 2,
        }));
      }
      const fromDot = el("circle", {
        cx: x1, cy: y, r: 4.5, fill: COLORS.line,
      });
      fromDot.appendChild(el("title", {}, [document.createTextNode(
        `${row.label} — current ${(opts.format || String)(row.from)}`
      )]));
      svg.appendChild(fromDot);
      const toDot = el("circle", {
        cx: x2, cy: y, r: 4.5,
        fill: row.to < row.from ? COLORS.accent : COLORS.line,
      });
      toDot.appendChild(el("title", {}, [document.createTextNode(
        `${row.label} — recommended ${(opts.format || String)(row.to)}`
      )]));
      svg.appendChild(toDot);
      svg.appendChild(textEl((opts.format || String)(row.to), {
        x: width - pad.right + 8, y: y + 4, "font-size": "10.5",
        fill: row.to < row.from ? COLORS.accent : COLORS.text,
      }));
    });

    container.classList.add("chart-box");
    container.innerHTML = "";
    container.appendChild(svg);
  }

  window.Charts = { lineChart, barChart, groupedBars, dumbbells, COLORS };
})();
