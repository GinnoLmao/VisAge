import { element } from "./dom.js";
let plotId = 0;
export const COLORS = ["#2563eb", "#0c9288", "#8557c5", "#cf8a27"];
function svgNode(tag, attributes = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text !== undefined) node.textContent = String(text); return node;
}
export function lineChart(container, series, { title, xLabel = "False positive rate", yLabel = "True positive rate", xMax = 1, baseline = false } = {}) {
  const safe = series.map((line, i) => ({ ...line, color: line.color || COLORS[i % COLORS.length], points: line.points.filter(point => Number.isFinite(point[0]) && Number.isFinite(point[1])) })).filter(line => line.points.length);
  const id = `plot-clip-${++plotId}`;
  function draw() {
    const width = Math.round(container.clientWidth); if (width < 120) return;
    if (!safe.length) { container.replaceChildren(element("p", "chart-caption", "No curve is available in this evidence record.")); return; }
    const height = 280, left = 49, top = 18, right = 15, bottom = 49, plotW = width - left - right, plotH = height - top - bottom;
    const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title, preserveAspectRatio: "xMidYMid meet" });
    svg.append(svgNode("title", {}, title), svgNode("desc", {}, `${yLabel} plotted against ${xLabel}. Exact numerical metrics are provided separately.`));
    const defs = svgNode("defs"), clip = svgNode("clipPath", { id }); clip.append(svgNode("rect", { x: left, y: top, width: plotW, height: plotH })); defs.append(clip); svg.append(defs);
    const x = value => left + value / xMax * plotW, y = value => top + (1 - value) * plotH;
    for (let i = 0; i <= 5; i += 1) {
      const fraction = i / 5;
      svg.append(svgNode("line", { x1: left, x2: left + plotW, y1: y(fraction), y2: y(fraction), stroke: "#edf1f7", "stroke-width": 1 }));
      const label = xMax === 1 ? fraction.toFixed(1) : (fraction * xMax).toFixed(xMax < 5 ? 1 : 0);
      svg.append(svgNode("text", { x: x(fraction * xMax), y: height - 29, "text-anchor": "middle", fill: "#596b84", stroke: "none", "font-size": 12 }, label));
      svg.append(svgNode("text", { x: left - 10, y: y(fraction) + 4, "text-anchor": "end", fill: "#596b84", stroke: "none", "font-size": 12 }, fraction.toFixed(1)));
    }
    svg.append(svgNode("line", { x1: left, y1: top, x2: left, y2: top + plotH, stroke: "#ced8e8", "stroke-width": 1 }));
    svg.append(svgNode("line", { x1: left, y1: top + plotH, x2: left + plotW, y2: top + plotH, stroke: "#ced8e8", "stroke-width": 1 }));
    if (baseline) svg.append(svgNode("path", { d: `M${x(0)},${y(0)}L${x(1)},${y(1)}`, stroke: "#bdc9da", "stroke-dasharray": "4 5", "stroke-width": 1.2, fill: "none" }));
    safe.forEach(line => {
      const path = line.points.map((point, i) => `${i ? "L" : "M"}${x(Math.max(0, Math.min(xMax, point[0]))).toFixed(2)},${y(Math.max(0, Math.min(1, point[1]))).toFixed(2)}`).join(" ");
      svg.append(svgNode("path", { d: path, stroke: line.color, "stroke-width": 2.2, fill: "none", "clip-path": `url(#${id})`, "stroke-linejoin": "round" }));
    });
    svg.append(svgNode("text", { x: left + plotW / 2, y: height - 3, "text-anchor": "middle", fill: "#596b84", stroke: "none", "font-size": 13 }, xLabel));
    svg.append(svgNode("text", { transform: `translate(13 ${top + plotH / 2}) rotate(-90)`, "text-anchor": "middle", fill: "#596b84", stroke: "none", "font-size": 13 }, yLabel));
    container.replaceChildren(svg);
  }
  const observer = new ResizeObserver(draw); observer.observe(container); draw();
  return () => observer.disconnect();
}
export function legend(series) {
  const node = element("div", "chart-legend");
  series.forEach((line, i) => { const item = element("span", "legend-item"), swatch = element("span", `legend-line legend-color-${i}`); item.append(swatch, document.createTextNode(line.label)); node.append(item); });
  return node;
}
