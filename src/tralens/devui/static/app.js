"use strict";
const $ = (id) => document.getElementById(id);
let config, current, snapshot, sourceImage, selected = null, timer, busy = false;
const colors = ["#ed384b", "#008bdd", "#009b55", "#bf4cd4", "#c37800"];
const node = (tag, text, className) => {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
};
const artifactUrl = (name) => `/api/jobs/${current}/artifacts/${name.split("/").map(encodeURIComponent).join("/")}`;
const showError = (message) => { $("error").textContent = message || ""; $("error").hidden = !message; };
async function request(url, settings) {
  const response = await fetch(url, settings);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
function setBusy(value) {
  busy = value;
  $("run").disabled = value || !config;
  $("jobs").disabled = value;
  for (const id of ["file", "mode", "budget", "policy", "fragments"]) $(id).disabled = value;
}
function drawSource() {
  if (!sourceImage || !sourceImage.complete || !sourceImage.naturalWidth || !snapshot) return;
  const canvas = $("source"), context = canvas.getContext("2d");
  canvas.width = sourceImage.naturalWidth;
  canvas.height = sourceImage.naturalHeight;
  context.drawImage(sourceImage, 0, 0);
  if ($("overlay").value === "none") return;
  const unitByMember = new Map();
  snapshot.units.forEach((unit, index) => unit.member_ids.forEach((id) => unitByMember.set(JSON.stringify(id), {unit, index})));
  snapshot.detections.forEach((detection, index) => {
    const owner = unitByMember.get(JSON.stringify(detection.source_id));
    const active = !selected || owner?.unit.id === selected;
    context.globalAlpha = active ? 1 : 0.2;
    context.strokeStyle = colors[($("overlay").value === "units" && owner ? owner.index : index) % colors.length];
    context.lineWidth = Math.max(2, canvas.width / 500);
    context.beginPath();
    detection.polygon.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
    context.closePath(); context.stroke();
    const [x, y] = detection.polygon[0];
    context.font = `${Math.max(14, canvas.width / 65)}px sans-serif`;
    context.fillStyle = context.strokeStyle;
    context.fillText($("overlay").value === "units" && owner ? `U${owner.index + 1}` : String(detection.source_id), x, Math.max(18, y - 4));
  });
  context.globalAlpha = 1;
}
function stages(data) {
  const order = ["upload", "ocr", "translation", "rendering"];
  let active = data.stage;
  if (active === "starting") active = "ocr";
  if (active === "finalizing" || active === "finished") active = data.options.mode === "render" ? "rendering" : data.options.mode === "translate" ? "translation" : "ocr";
  if (data.error) active = data.error.stage === "rendering" ? "rendering" : data.preview_stage === "translation" ? "translation" : "ocr";
  for (const element of $("stages").children) {
    const stage = element.dataset.stage;
    const skipped = (stage === "translation" && data.options.mode === "ocr") || (stage === "rendering" && data.options.mode !== "render");
    let state = skipped ? "skipped" : order.indexOf(stage) < order.indexOf(active) ? "complete" : "pending";
    if (!skipped && stage === active) state = data.status === "running" ? "running" : data.error ? "failed" : "complete";
    if (stage === "rendering" && data.rendering_skipped) state = "skipped";
    element.dataset.state = state;
    element.textContent = ({upload: "Upload", ocr: "OCR + grouping", translation: "Translation", rendering: "Rendering"})[stage] + ` · ${state}`;
  }
}
function renderUnits() {
  $("units").replaceChildren();
  snapshot.units.forEach((unit, index) => {
    const box = node("article", undefined, "unit" + (unit.id === selected ? " selected" : ""));
    const highlight = node("button", `U${index + 1} · ${unit.status}`);
    highlight.type = "button";
    highlight.addEventListener("click", () => { selected = unit.id; drawSource(); renderUnits(); });
    box.append(highlight, node("small", `  Sources: ${unit.member_ids.join(", ")}`), node("pre", unit.source || "No selected source reading"));
    if (unit.target) box.append(node("pre", unit.target, "target"));
    if (unit.reasons.length) box.append(node("div", unit.reasons.join("; "), "note"));
    for (const fragment of unit.fragments) {
      box.append(node("div", `Fragment (review-only): ${fragment.status}`, "note"));
      for (const [, text] of fragment.translations) box.append(node("pre", text));
    }
    if (unit.render_status) box.append(node("div", `Render: ${unit.render_status}${unit.quality ? " · " + unit.quality : ""}${unit.render_reason ? " · " + unit.render_reason : ""}`));
    if (unit.warnings.length) box.append(node("div", unit.warnings.join("\n"), "warning"));
    $("units").append(box);
  });
  if (!snapshot.units.length) $("units").append(node("p", snapshot.status === "running" ? "Waiting for OCR evidence…" : "No units produced."));
}
function layers(data) {
  for (const [container, choices] of [
    ["layers", [["clean", "Cleanplate"], ["text", "Translated text layer"], ["composite", "Composite"]]],
    ["masks", [["erase-mask", "Erase support"], ["alpha-mask", "Text alpha"], ["protected-mask", "Protected pixels"]]]
  ]) {
    const parent = $(container);
    for (const [name, title] of choices) {
      const path = `run/rendering/${name}.png`;
      if (!data.artifacts.includes(path) || parent.querySelector(`[data-layer="${name}"]`)) continue;
      const figure = node("figure"); figure.dataset.layer = name;
      const image = node("img"); image.src = artifactUrl(path); image.alt = title;
      const link = node("a"); link.href = image.src; link.target = "_blank"; link.rel = "noopener";
      link.append(image); figure.append(node("figcaption", title), link); parent.append(figure);
    }
  }
  $("artifacts").replaceChildren();
  for (const path of data.artifacts) {
    const li = node("li"), link = node("a", path);
    link.href = artifactUrl(path); link.target = "_blank"; link.rel = "noopener";
    li.append(link); $("artifacts").append(li);
  }
}
function display(data) {
  snapshot = data;
  $("results").hidden = false;
  $("status").textContent = `${data.status} · ${data.stage} · ${data.elapsed_seconds}s`;
  const translations = data.units.filter((u) => u.status === "preview" && u.target).length;
  $("summary").textContent = `${data.detections.length} raw detections · ${data.units.length} units · ${translations} strict translation previews · ${data.model_calls} model calls (last checkpoint)`;
  $("output-directory").textContent = `Artifacts: ${data.output_directory}`;
  $("notices").textContent = data.notices.join("\n");
  $("ocr-text").textContent = data.detections.map((d) => `${d.source_id}: ${d.text} (confidence: ${d.confidence ?? "unknown"})`).join("\n");
  const quality = data.quality;
  $("render-note").textContent = data.rendering_skipped ? `Rendering skipped: ${data.rendering_skipped}` : data.rendering_coverage
    ? `${data.rendering_coverage.rendered}/${data.rendering_coverage.saved_units} saved units rendered; ${quality?.best_effort_rendered || 0} best-effort. ${data.rendering_coverage.partial_output ? "Partial output." : "Candidate output only."}`
    : "Layers appear only when rendering actually produces them.";
  showError(data.error ? (data.error.message || String(data.error)) : "");
  stages(data); renderUnits(); layers(data);
  if (!sourceImage) {
    sourceImage = new Image(); sourceImage.addEventListener("load", drawSource);
    sourceImage.src = artifactUrl(data.input);
  } else drawSource();
  setBusy(data.status === "running");
}
async function poll(identity) {
  try {
    const data = await request(`/api/jobs/${identity}`);
    if (identity !== current) return;
    display(data);
    if (data.status === "running") timer = setTimeout(() => poll(identity), 1000);
  } catch (error) {
    if (identity !== current) return;
    showError(`Could not refresh: ${error.message}. Retrying…`);
    timer = setTimeout(() => poll(identity), 2000);
  }
}
function selectJob(identity) {
  clearTimeout(timer); current = identity; sourceImage = null; selected = null;
  $("results").hidden = true;
  $("layers").replaceChildren(); $("masks").replaceChildren();
  poll(identity);
}
$("run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("file").files[0];
  if (!file || busy) return;
  if (file.size > config.max_upload_bytes) return showError("Upload limit is 32 MiB.");
  const settings = {mode: $("mode").value, max_attempts: Number($("budget").value), allow_fragments: $("fragments").checked, render_policy: $("policy").value};
  setBusy(true); showError(""); $("status").textContent = "Uploading…";
  try {
    const result = await request("/api/jobs", {method: "POST", headers: {"Content-Type": "application/octet-stream", "X-Tralens-Token": config.token, "X-Tralens-Options": JSON.stringify(settings)}, body: file});
    const choice = node("option", `${file.name} · ${settings.mode} · ${result.id.slice(0, 8)}`); choice.value = result.id;
    $("jobs").append(choice); $("jobs").value = result.id;
    selectJob(result.id);
  } catch (error) { showError(error.message); $("status").textContent = "Run was not started."; setBusy(false); }
});
$("jobs").addEventListener("change", () => { if ($("jobs").value) selectJob($("jobs").value); });
$("overlay").addEventListener("change", drawSource);
$("clear-selection").addEventListener("click", () => { selected = null; drawSource(); if (snapshot) renderUnits(); });
request("/api/config").then(async (value) => {
  config = value;
  $("configuration").textContent = `Local only · ${config.device} · ${config.model} · OCR exports: ${config.model_root}${config.fonts ? " · Font manifest: " + config.fonts : " · No --fonts: rendering disabled"}`;
  $("mode").querySelector('[value="render"]').disabled = !config.render_enabled;
  $("mode").value = config.render_enabled ? "render" : "ocr";
  setBusy(false);
  const jobs = await request("/api/jobs");
  for (const job of jobs) {
    const choice = node("option", `${job.id.slice(0, 8)} · ${job.mode} · ${job.status}`); choice.value = job.id;
    $("jobs").append(choice);
  }
  if (jobs.length) {
    const last = jobs[jobs.length - 1];
    $("jobs").value = last.id; selectJob(last.id);
  }
}).catch((error) => showError(error.message));
