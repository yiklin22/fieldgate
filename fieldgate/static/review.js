"use strict";
const $ = (id) => document.getElementById(id);
let current = null, selected = null;

async function api(path, body) {
  const options = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) k === "class" ? (node.className = v) : k.startsWith("on") ? node.addEventListener(k.slice(2), v) : node.setAttribute(k, v);
  for (const child of children) node.append(child);
  return node;
}

function showMonitor(m) {
  const node = $("alarm");
  node.className = m.alarm ? "on" : "";
  node.textContent = m.alarm ? `audit alarm: ${m.errors} wrong in ${m.audited} checked. Stop auto-approving and re-certify.`
                             : `audit: ${m.errors} wrong in ${m.audited} checked`;
}

async function loadQueue(keep) {
  const q = await api("/api/queue");
  $("s-approved").textContent = `${q.approved} of ${q.fields} fields`;
  $("s-touchless").textContent = `${q.touchless} of ${q.docs.length}`;
  $("s-target").textContent = `≤ ${Math.round(q.alpha * 100)}% wrong among approved, ${Math.round((1 - q.delta) * 100)}% confidence`;
  showMonitor(q.monitor);
  const list = $("docs");
  list.replaceChildren(...q.docs.map((d) => el("button", { class: current && d.id === current.id ? "sel" : "", onclick: () => openDoc(d.id) },
    el("span", {}, d.id), el("span", { class: "n" + (d.to_review ? "" : " zero") }, d.to_review ? `${d.to_review} to check` : "done"))));
  if (!keep) { const first = q.docs.find((d) => d.to_review) || q.docs[0]; if (first) openDoc(first.id); }
}

function drawBoxes(field) {
  const svg = $("overlay");
  svg.setAttribute("viewBox", `0 0 ${current.width} ${current.height}`);
  svg.replaceChildren(...(field ? current.fields[field].boxes : []).map(([x, y, w, h]) => {
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", x - 4); r.setAttribute("y", y - 4); r.setAttribute("width", w + 8); r.setAttribute("height", h + 8); r.setAttribute("rx", 4);
    return r;
  }));
}

function renderFields() {
  const cards = Object.entries(current.fields).map(([name, f]) => {
    const needs = !f.approved || f.audit;
    const chip = f.decision ? el("span", { class: "chip done" }, f.decision.action === "correct" ? "corrected" : "checked")
      : !f.approved ? el("span", { class: "chip review" }, "needs review")
      : f.audit ? el("span", { class: "chip audit" }, "approved · audit sample") : el("span", { class: "chip ok" }, "auto-approved");
    const shown = f.decision ? f.decision.value : f.value;
    const card = el("div", { class: "field" + (name === selected ? " sel" : ""), onclick: () => { selected = name; renderFields(); drawBoxes(name); } },
      el("div", { class: "top" }, el("span", { class: "name" }, name), chip),
      el("div", { class: "top" }, el("span", { class: "value" }, shown === null ? "not on receipt" : String(shown)),
        el("span", { class: "muted" }, `score ${f.score.toFixed(2)}`)));
    if (f.why.length && !f.approved) card.append(el("ul", { class: "why" }, ...f.why.map((w) => el("li", {}, w))));
    if (needs && !f.decision) {
      const input = el("input", { type: "text", placeholder: "correct value", "aria-label": `Correct value for ${name}`, onclick: (e) => e.stopPropagation() });
      const error = el("div", { class: "err" });
      const send = async (e, body) => { e.stopPropagation(); try { const out = await api(`/api/doc/${current.id}/${name}`, body); showMonitor(out.monitor); await openDoc(current.id); loadQueue(true); } catch (err) { error.textContent = err.message; } };
      card.append(el("div", { class: "act" },
        el("button", { class: "primary", onclick: (e) => send(e, { action: "accept" }) }, "Value is right"),
        input, el("button", { onclick: (e) => send(e, { action: "correct", value: input.value.trim() || null }) }, input.value ? "Save" : "Save correction")), error);
    }
    return card;
  });
  $("fields").replaceChildren(...cards);
}

async function openDoc(id) {
  current = await api(`/api/doc/${id}`);
  $("img").src = `/image/${id}`;
  const pending = Object.entries(current.fields).find(([, f]) => (!f.approved || f.audit) && !f.decision);
  selected = pending ? pending[0] : null;
  renderFields();
  drawBoxes(selected);
  for (const b of $("docs").children) b.classList.toggle("sel", b.firstChild.textContent === id);
}

loadQueue(false);
