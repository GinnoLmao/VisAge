export function element(tag, className = "", text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
export function icon(name, className = "icon") {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  node.setAttribute("class", className); node.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(node.namespaceURI, "use"); use.setAttribute("href", `#i-${name}`); node.append(use); return node;
}
export function button(label, className = "button secondary", action) {
  const node = element("button", className, label); node.type = "button";
  if (action) node.addEventListener("click", action);
  return node;
}
export function panel(title, subtitle, label) {
  const node = element("section", "panel"), head = element("div", "panel-heading"), words = element("div");
  words.append(element("h2", "", title)); if (subtitle) words.append(element("p", "", subtitle)); head.append(words);
  if (label) head.append(element("span", "evidence-label", label)); node.append(head); return node;
}
export function note(text) { const node = element("div", "scope-note"); node.append(icon("info"), element("p", "", text)); return node; }
