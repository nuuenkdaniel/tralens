"use strict";

// Pure view-state helpers are also exercised by the optional Node.js tests.
const VIEW_ORDER = ["original", "ocr", "translation", "clean", "final"];
const OUTPUTS = {clean: "run/rendering/clean.png", final: "run/rendering/composite.png"};
function outputReady(data, view) {
  return Boolean(data?.rendering_coverage && data.artifacts?.includes(OUTPUTS[view]));
}
function previewStates(data, hasSource = false) {
  const states = Object.fromEntries(VIEW_ORDER.map(view => [view, {state: "waiting", label: "Waiting", available: false}]));
  states.original = {state: hasSource ? "ready" : "waiting", label: hasSource ? "Source image" : "Choose an image", available: hasSource};
  if (!data) return states;
  const running = data.status === "running";
  const ocrAvailable = Boolean(data.detections.length || data.units.length || ["translation", "finished"].includes(data.preview_stage));
  const failedAt = data.error ? (data.error.stage === "rendering" ? "rendering" : data.preview_stage === "translation" ? "translation" : data.preview_stage ? "ocr" : "startup") : null;
  const upstreamFailed = ["startup", "ocr"].includes(failedAt);
  states.ocr = {state: failedAt === "startup" ? "skipped" : failedAt === "ocr" ? "failed" : ocrAvailable ? "ready" : running ? "running" : "waiting",
    label: failedAt === "startup" ? "Setup failed" : failedAt === "ocr" ? "OCR failed" : ocrAvailable ? `${data.detections.length} text region${data.detections.length === 1 ? "" : "s"}` : running ? data.stage === "starting" ? "Starting…" : "Finding text…" : "No OCR output", available: ocrAvailable};
  const count = data.units.filter(unit => unit.status === "preview" && unit.target).length;
  const translating = running && data.stage === "translation";
  states.translation = {state: data.options.mode === "ocr" || upstreamFailed ? "skipped" : failedAt === "translation" ? "failed" : translating ? "running" : ocrAvailable ? "ready" : "waiting",
    label: data.options.mode === "ocr" ? "Not requested" : upstreamFailed ? "OCR did not finish" : failedAt === "translation" ? "Translation failed" : translating ? `${count} ready · translating…` : count ? `${count} text preview${count === 1 ? "" : "s"}` : ocrAvailable && !running ? "No translations" : "Waiting for text",
    available: ocrAvailable && data.options.mode !== "ocr"};
  for (const view of ["clean", "final"]) {
    const available = outputReady(data, view);
    const skipped = data.options.mode !== "render" || Boolean(data.rendering_skipped) || upstreamFailed || failedAt === "translation";
    states[view] = {state: available ? "ready" : skipped ? "skipped" : failedAt === "rendering" ? "failed" : running && ["rendering", "finalizing"].includes(data.stage) ? "running" : "waiting",
      label: available ? (view === "clean" ? "Cleanup layer" : data.display_summary ? `${data.display_summary.displayed_translations} displayed` : `${data.rendering_coverage.rendered} replacement${data.rendering_coverage.rendered === 1 ? "" : "s"}`) : skipped ? "Not produced" : failedAt === "rendering" ? "Rendering failed" : running && ["rendering", "finalizing"].includes(data.stage) ? "Rendering…" : "Waiting for translation", available};
  }
  return states;
}
function progressView(data) {
  if (outputReady(data, "final")) return "final";
  if (data.stage === "rendering" || data.stage === "finalizing") return "final";
  if (data.options.mode !== "ocr" && ["translation", "finished"].includes(data.preview_stage)) return "translation";
  return "ocr";
}
function missingOutputMessage(data, view) {
  if (!data) return {title: "This stage hasn't run yet", copy: "Click Run pipeline to see this stage. Your original image is ready to inspect."};
  if ((view === "clean" || view === "final") && data.rendering_skipped === "no_strict_translations") {
    return {title: "No rendered page was produced", copy: "No complete, source-bound translations were available. Open Translation to see which regions were blocked, failed, or reached the call budget. The original is unchanged."};
  }
  if ((view === "clean" || view === "final") && data.options.mode !== "render") return {title: "Rendering wasn't requested", copy: "This run stopped before rendering. Choose Final rendering in Run settings and start a new run to create the cleaned and final pages."};
  if (view === "translation" && data.options.mode === "ocr") return {title: "Translation wasn't requested", copy: "This was an OCR-only run. Choose Translation or Final rendering in Run settings to go further."};
  if (data.error || data.status === "failed" || data.status === "interrupted") return {title: "This stage didn't produce an output", copy: "The run stopped before this output was published. Earlier stage results and the original remain available. See the error above for details."};
  return {title: data.status === "running" ? "Waiting for this stage…" : "No output for this stage", copy: data.status === "running" ? "The real output will appear here when the pipeline publishes it. You can inspect earlier stages while you wait." : "This run did not publish this output. Inspect Translation and the run details to see why."};
}
function qualityNotice(data, rereadAvailable = false) {
  if (!data || !outputReady(data, "final")) return null;
  if (data.visibility_policy === "always-show" || data.options?.visibility_policy === "always-show") {
    const counts = data.display_summary;
    return {copy: "Prioritizes visible translations; fallback patches may cover artwork. " +
      (counts ? `${counts.clean_replacements} clean replacements · ${counts.patch_backed_translations} patch-backed translations · ${counts.nearby_captions} nearby captions · ${counts.intentionally_unchanged} intentionally unchanged · ${counts.available_not_displayed} available but not displayed · ${counts.missing_translations} missing translations. ` : "") +
      "Captions retain Japanese source text. Patches and captions are not artwork reconstruction or human-verified translations.", action: null};
  }
  if (data.replacement_policy !== "confidence-gated") return {
    copy: "This run used broad replacement without artwork checks. It can smear drawings and repaint bad OCR. Retry with native cleanup / OCR checks; unsupported edits will stay original, with translations retained.",
    action: rereadAvailable ? "Retry with quality checks" : "Retry with artwork protection"
  };
  const counts = data.replacement_confidence;
  return {copy: `Experimental native-cleanup / OCR checks used${data.quality_reread?.enabled ? " with paired rereads" : " without paired rereads"}. ` +
    (counts ? `${counts.actual_replacements} image replacements; ${counts.bound_translations_without_replacement} translations kept without repainting. ` : "") +
    "Untranslated text and wrong readings may remain. This is an unverified candidate—not a finished translated page.", action: null};
}
function localizationNotice(data) {
  const crop = data?.source_localization;
  if (!crop?.enabled) return "";
  const unknown = data.status === "running" ? "pending" : "not recorded";
  return ` Experimental crop localization: ${crop.requested ?? unknown} requested, ${crop.calls ?? unknown} calls, ${crop.predictions_consumed ?? unknown} predictions reaching cleanup, ${crop.budget_missed ?? unknown} budget exclusions. Model agreement does not certify correctness.` +
    (crop.error ? ` Crop pass failed: ${crop.error}` : "") +
    (crop.failed ? ` ${crop.failed} crop predictions failed.` : "");
}
if (typeof module !== "undefined" && module.exports) module.exports = {previewStates, progressView, outputReady, missingOutputMessage, qualityNotice, localizationNotice};

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const colors = ["#157eac", "#9562b5", "#c07932", "#25845e", "#c65568"];
  const titles = {original: "Original page", ocr: "Detected text", translation: "Translation previews", clean: "Cleaned page", final: "Rendered candidate"};
  const descriptions = {
    original: "Your original image, exactly as uploaded.", ocr: "Text regions found by OCR. Select a region on the page or in the list.",
    translation: "Translated regions are highlighted. Read the English text alongside the original page.",
    clean: "The actual cleanup layer, before translated lettering is added.", final: "The actual composite: cleanup and translated lettering together."
  };
  let config, current = null, snapshot = null, sourceImage = null, sourceURL = null, settingsJob = null;
  let selected = null, selectedView = "original", timer, busy = false, pendingFile = null, localURL = null;
  const outputImages = new Map();
  const node = (tag, text, className) => {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    if (className) result.className = className;
    return result;
  };
  const sourceLoaded = () => Boolean(sourceImage?.complete && sourceImage.naturalWidth);
  const artifactUrl = name => `/api/jobs/${current}/artifacts/${name.split("/").map(encodeURIComponent).join("/")}`;
  const showError = message => { $("error").textContent = message || ""; $("error").hidden = !message; };
  const friendly = value => (value || "").replaceAll("_", " ");
  async function request(url, settings) {
    const response = await fetch(url, settings);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    return result;
  }
  function setBusy(value) {
    busy = value;
    const retry = Boolean(current && snapshot && snapshot.status !== "running");
    $("run").disabled = value || !config || (!pendingFile && !retry);
    $("run").textContent = value ? "Running…" : retry ? "Retry pipeline ↻" : "Run pipeline →";
    $("run").title = retry ? "Rerun this saved upload with the current settings. Previous results are kept." : "Start a new run with the selected image.";
    $("jobs").disabled = value;
    for (const id of ["file", "mode", "budget", "policy", "translation-policy", "fragments"]) $(id).disabled = value;
    $("replacement-policy").disabled = value || $("mode").value !== "render";
    $("quality-reread").disabled = value || !config?.quality_reread_available || $("mode").value !== "render" || $("translation-policy").value !== "coverage-first";
    $("layout-policy").disabled = value || $("mode").value !== "render";
    $("source-localization").disabled = value || $("mode").value !== "render";
    $("visibility-policy").disabled = value || $("mode").value !== "render";
    $("quality-retry").disabled = value || !config?.render_enabled;
  }
  function settingsSummary() {
    const mode = {ocr: "OCR only", translate: "Through translation", render: "Through final rendering"}[$("mode").value];
    $("settings-summary").textContent = mode + ($("mode").value === "ocr" ? " · no model calls" : ` · ${$("translation-policy").value === "coverage-first" ? "coverage first" : "conservative translation"} · up to ${$("budget").value} translation calls`);
    if ($("mode").value === "render") $("settings-summary").textContent += $("replacement-policy").value === "confidence-gated" ? " · artwork checks" : " · broad replacement";
    if ($("mode").value === "render" && $("layout-policy").value === "enclosed-retry") $("settings-summary").textContent += " · enclosure retry";
    if ($("mode").value === "render" && $("source-localization").checked) $("settings-summary").textContent += " · requested crop OCR";
    if ($("mode").value === "render" && $("visibility-policy").value === "always-show") $("settings-summary").textContent += " · visible translations / artwork may be covered";
    setBusy(busy);
  }
  function ownership() {
    const result = new Map();
    snapshot?.units.forEach((unit, index) => unit.member_ids.forEach(id => result.set(JSON.stringify(id), {unit, index})));
    return result;
  }
  function drawOverlay(context, kind, highlight = true) {
    if (!snapshot || !["ocr", "translation"].includes(kind)) return;
    const owners = ownership();
    const width = Math.max(2, sourceImage.naturalWidth / 480), fontSize = Math.max(14, sourceImage.naturalWidth / 75);
    snapshot.detections.forEach((detection, index) => {
      const owner = owners.get(JSON.stringify(detection.source_id));
      const translated = owner?.unit.status === "preview" && Boolean(owner.unit.target);
      const color = kind === "translation" ? (translated ? "#14845e" : "#8e9f95") : colors[(owner ? owner.index : index) % colors.length];
      context.globalAlpha = !highlight || !selected || owner?.unit.id === selected ? 1 : .2;
      context.lineWidth = width;
      context.strokeStyle = color;
      context.fillStyle = color + "14";
      context.setLineDash(kind === "translation" && !translated ? [width * 2, width * 2] : []);
      context.beginPath();
      detection.polygon.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
      context.closePath(); context.fill(); context.stroke(); context.setLineDash([]);
      const label = $("overlay").value === "detections" && kind === "ocr" ? String(detection.source_id) : owner ? String(owner.index + 1) : "?";
      const x = detection.polygon[0][0], y = Math.max(fontSize + 5, detection.polygon[0][1]);
      context.font = `600 ${fontSize}px system-ui, sans-serif`;
      const labelWidth = context.measureText(label).width + 10;
      context.fillStyle = color; context.fillRect(x, y - fontSize - 6, labelWidth, fontSize + 6);
      context.fillStyle = "#ffffff"; context.fillText(label, x + 5, y - 5);
    });
    context.globalAlpha = 1;
  }
  function paint(canvas, image, kind, thumbnail = false) {
    const context = canvas.getContext("2d");
    canvas.width = thumbnail ? 260 : image.naturalWidth;
    canvas.height = thumbnail ? 110 : image.naturalHeight;
    context.save();
    if (thumbnail) {
      const scale = Math.min(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
      context.translate((canvas.width - image.naturalWidth * scale) / 2, (canvas.height - image.naturalHeight * scale) / 2);
      context.scale(scale, scale);
    }
    context.drawImage(image, 0, 0);
    drawOverlay(context, kind, !thumbnail);
    context.restore();
  }
  function getOutputImage(view) {
    if (!outputReady(snapshot, view)) return null;
    const path = OUTPUTS[view];
    if (!outputImages.has(path)) {
      const image = new Image(), identity = current;
      outputImages.set(path, image);
      image.addEventListener("load", () => { if (identity === current) { updateThumbnails(); renderImage(); } });
      image.addEventListener("error", () => { if (identity === current) showError("Could not load a published image. Refresh the page to retry."); });
      image.src = artifactUrl(path);
    }
    return outputImages.get(path);
  }
  function updateThumbnails() {
    const states = previewStates(snapshot, sourceLoaded());
    for (const view of VIEW_ORDER) {
      const canvas = $("thumb-" + view);
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      if (view === "original" && sourceLoaded()) paint(canvas, sourceImage, view, true);
      if (["ocr", "translation"].includes(view) && states[view].available && sourceLoaded()) paint(canvas, sourceImage, view, true);
      if (["clean", "final"].includes(view)) {
        const image = getOutputImage(view);
        if (image?.complete && image.naturalWidth) paint(canvas, image, view, true);
      }
      const button = document.querySelector(`#stages [data-stage="${view}"]`);
      button.dataset.state = states[view].state;
      button.setAttribute("aria-pressed", String(view === selectedView));
      button.querySelector(".stage-status").textContent = states[view].label;
    }
  }
  function renderImage() {
    const states = previewStates(snapshot, sourceLoaded());
    const available = states[selectedView].available;
    $("view-title").textContent = titles[selectedView];
    $("view-description").textContent = descriptions[selectedView];
    $("image-caption").textContent = selectedView === "translation" ? "Original + translation highlights" : titles[selectedView];
    $("source").hidden = true; $("stage-image").hidden = true; $("stage-placeholder").hidden = true;
    $("download").hidden = true;
    const comparison = available && sourceLoaded() && selectedView !== "original";
    $("compare").disabled = !comparison;
    $("comparison").hidden = !comparison || !$("compare").checked;
    if (comparison && $("compare").checked && $("compare-source").src !== sourceImage.src) $("compare-source").src = sourceImage.src;
    if (available && ["original", "ocr", "translation"].includes(selectedView) && sourceLoaded()) {
      $("source").hidden = false;
      $("source").dataset.interactive = String(selectedView !== "original");
      paint($("source"), sourceImage, selectedView);
      if (selectedView === "original") { $("download").href = sourceImage.src; $("download").hidden = false; }
    } else if (available && ["clean", "final"].includes(selectedView)) {
      const image = getOutputImage(selectedView);
      if (image?.complete && image.naturalWidth) {
        if ($("stage-image").src !== image.src) $("stage-image").src = image.src;
        $("stage-image").hidden = false; $("stage-image").alt = titles[selectedView];
        $("download").href = image.src; $("download").hidden = false;
      }
    } else {
      const message = missingOutputMessage(snapshot, selectedView);
      $("stage-placeholder").hidden = false;
      $("placeholder-title").textContent = message.title;
      $("placeholder-copy").textContent = message.copy;
      $("show-previous").textContent = selectedView === "final" || selectedView === "clean" ? "View translations" : "View original";
    }
    $("image-note").textContent = selectedView === "translation" ? "Translation is text-only at this stage. Green regions have English previews; dashed regions do not. Lettering is placed in Final."
      : selectedView === "ocr" ? "These are inspection overlays, not erase masks. Click a region to highlight its text."
      : selectedView === "clean" ? "Only the published cleanup image is shown. Best-effort reconstruction may alter artwork."
      : selectedView === "final" ? "This is the actual published composite, including any untranslated or skipped source regions."
      : "Your original file is preserved. Previews do not change the pipeline inputs.";
  }
  function unitStatus(unit) {
    if (selectedView === "ocr") return unit.status === "blocked" ? ["Needs review", "warning"] : ["Recognized", "ready"];
    if (selectedView === "final" && unit.display_status === "patch") return ["Opaque patch", "warning"];
    if (selectedView === "final" && unit.display_status === "caption") return ["Nearby caption · source retained", "warning"];
    if (["clean", "final"].includes(selectedView) && unit.render_status) {
      if (unit.render_status === "automatic_composed") return [unit.quality === "best_effort" ? "Best effort" : selectedView === "clean" ? "Cleaned" : "Replaced", unit.quality === "best_effort" ? "warning" : "ready"];
      return ["Not replaced", "warning"];
    }
    if (unit.status === "preview" && unit.target) return unit.input_kind === "coverage_text_region" ? ["Translated · uncertain", "warning"] : ["Translated", "ready"];
    if (unit.status.includes("budget")) return ["Call budget reached", "warning"];
    if (unit.status === "blocked") return ["Blocked by OCR", "warning"];
    if (unit.status.includes("failed")) return ["Translation failed", "failed"];
    if (unit.status === "pending") return ["Waiting", "muted"];
    if (unit.status === "ocr_only") return ["Not requested", "muted"];
    return [friendly(unit.status), "muted"];
  }
  function chooseUnit(identity, scroll = false) {
    selected = identity; renderInspector(); renderImage();
    if (scroll) $("units").querySelector(".selected")?.scrollIntoView({block: "nearest", behavior: "smooth"});
  }
  function renderInspector() {
    const list = $("units"), scroll = list.scrollTop;
    const open = new Set([...list.querySelectorAll(".unit details[open]")].map(details => details.parentElement.dataset.unit));
    list.replaceChildren();
    $("overlay-controls").hidden = !["ocr", "translation"].includes(selectedView);
    $("unit-count").textContent = selectedView === "original" ? "Source" : String(snapshot?.units.length || 0);
    $("inspector-title").textContent = {original: "About this page", ocr: "Recognized text", translation: "Source → English", clean: "Cleanup results", final: "Replacement results"}[selectedView];
    $("inspector-note").textContent = {original: "Choose a stage above to see what changes.", ocr: "Numbers match the regions on the page.", translation: "English previews are shown below each original reading.", clean: "The page is shown before new lettering is added.", final: "See which regions were replaced, and which stayed original."}[selectedView];
    if (selectedView === "ocr" && $("overlay").value === "detections") $("inspector-note").textContent = "Raw OCR box IDs are shown on the page. This list groups them into text units.";
    if (selectedView === "original") {
      const size = snapshot?.size || (sourceLoaded() ? [sourceImage.naturalWidth, sourceImage.naturalHeight] : null);
      const facts = node("dl", undefined, "page-facts");
      for (const [label, value] of [["Dimensions", size ? size.join(" × ") + " px" : "Loading…"], ["Processing", "Local only"], ["Original", "Preserved"]]) facts.append(node("dt", label), node("dd", value));
      list.append(facts, node("p", snapshot ? "The stage previews use this exact uploaded page. Click Detected text to inspect OCR, or Final to see the composite if one was produced." : "Ready to start. Click Run pipeline and the stage thumbnails will fill in as real outputs become available.", "inspector-message"));
      return;
    }
    if (!snapshot?.units.length) {
      list.append(node("p", snapshot?.status === "running" ? "Waiting for the OCR checkpoint. Text regions will appear here when available." : "No text units are available for this stage.", "inspector-message"));
      return;
    }
    for (const [index, unit] of snapshot.units.entries()) {
      const box = node("article", undefined, "unit" + (unit.id === selected ? " selected" : "")); box.dataset.unit = unit.id;
      const heading = node("div", undefined, "unit-heading"), button = node("button", `Region ${index + 1}`);
      button.type = "button"; button.addEventListener("click", () => chooseUnit(unit.id));
      const [text, tone] = unitStatus(unit), badge = node("span", text, "unit-status"); badge.dataset.tone = tone;
      heading.append(button, badge); box.append(heading, node("pre", unit.source || "No selected source reading"));
      if (unit.reread_selected) {
        box.append(node("p", "Unverified paired reread used for translation; original OCR retained below.", "note"));
        const original = node("details"); original.append(node("summary", "Original OCR"), node("pre", unit.raw_source)); box.append(original);
      }
      if (["translation", "final"].includes(selectedView)) {
        if (unit.target) box.append(node("pre", unit.target, "target"));
        else if (selectedView === "translation") box.append(node("p", unit.status.includes("budget") ? "Not attempted: increase the call budget for a new run." : unit.status === "blocked" ? "OCR grouping or source ownership needs review; no complete translation was produced." : "No English preview for this region.", "note"));
      }
      if (unit.input_kind === "coverage_text_region" && unit.target) box.append(node("p", "Coverage-first OCR region: ownership, reading order and sentence completeness are not verified.", "note"));
      if (["clean", "final"].includes(selectedView) && unit.warnings.length) box.append(node("p", unit.warnings.join("\n"), "warning"));
      if (unit.reasons.length || unit.render_reason || unit.fragments.length || unit.crop_request_reason || unit.enclosure_layout_attempt || unit.display_fallback) {
        const details = node("details"); details.open = open.has(unit.id); details.append(node("summary", "Why / details"));
        details.append(node("p", `Source boxes: ${unit.member_ids.join(", ")} · ${unit.status}`));
        for (const reason of unit.reasons) details.append(node("p", friendly(reason)));
        if (unit.render_reason) details.append(node("p", unit.render_reason));
        if (unit.crop_request_reason) details.append(node("p", `Crop localization: ${friendly(unit.crop_request_status)} · ${friendly(unit.crop_request_reason)}`));
        for (const [label, attempt] of [["Localized cleanup", unit.localized_cleanup_attempt], ["Enclosure lettering", unit.enclosure_layout_attempt]]) {
          if (attempt) details.append(node("p", `${label}: ${friendly(attempt.status)}${attempt.reason ? " · " + attempt.reason : ""}`));
        }
        if (unit.display_fallback) {
          const fallback = unit.display_fallback;
          details.append(node("p", `Visible translation: ${friendly(fallback.kind || fallback.status)}${fallback.reason ? " · " + fallback.reason : ""}. Original source: ${friendly(unit.source_visibility)}.`));
          if (fallback.layout) details.append(node("p", `Exact saved target at ${fallback.layout.size}px. Backing/leader footprint: ${fallback.footprint_pixels} pixels; artwork may be covered.`));
          for (const [label, name] of [["Patch / caption backing", fallback.patch_artifact], ["Fallback text", fallback.text_artifact], ["Declared footprint", fallback.footprint_artifact]]) {
            if (!name || !snapshot.artifacts.includes("run/rendering/" + name)) continue;
            const link = node("a", label); link.href = artifactUrl("run/rendering/" + name); link.target = "_blank"; link.rel = "noopener";
            details.append(node("p")); details.lastChild.append(link);
          }
        }
        for (const fragment of unit.fragments) {
          details.append(node("p", `Review-only fragment: ${friendly(fragment.status)}`));
          for (const [, translation] of fragment.translations) details.append(node("pre", translation));
        }
        box.append(details);
      }
      list.append(box);
    }
    list.scrollTop = scroll;
  }
  function renderArtifacts(data) {
    for (const [container, choices] of [["layers", [["clean", "Cleanplate"], ["patch", "Opaque patches / caption backing"], ["text", "Translated text layer"], ["composite", "Composite"]]], ["masks", [["erase-mask", "Clean erasure support"], ["fallback-mask", "Declared fallback footprints"], ["alpha-mask", "Text alpha"], ["protected-mask", "Protected pixels"]]]]) {
      for (const [name, title] of choices) {
        const path = `run/rendering/${name}.png`;
        if (!data.rendering_coverage || !data.artifacts.includes(path) || $(container).querySelector(`[data-layer="${name}"]`)) continue;
        const figure = node("figure"), image = node("img"), link = node("a"); figure.dataset.layer = name;
        image.src = artifactUrl(path); image.alt = title; image.loading = "lazy";
        link.href = image.src; link.target = "_blank"; link.rel = "noopener"; link.append(image);
        figure.append(node("figcaption", title), link); $(container).append(figure);
      }
    }
    $("artifacts").replaceChildren();
    for (const path of data.artifacts) {
      const item = node("li"), link = node("a", path); link.href = artifactUrl(path); link.target = "_blank"; link.rel = "noopener"; item.append(link); $("artifacts").append(item);
    }
    $("output-directory").textContent = `Artifacts: ${data.output_directory}`;
    $("notices").textContent = data.notices.join("\n");
    $("ocr-text").textContent = data.detections.map(d => `${d.source_id}: ${d.text} (confidence: ${d.confidence ?? "unknown"})`).join("\n");
  }
  function selectView(view, manual = true) {
    selectedView = view;
    if (manual) $("follow").checked = false;
    updateThumbnails(); renderImage(); renderInspector();
  }
  function loadSource(url) {
    if (sourceURL === url) return;
    sourceURL = url;
    const image = new Image(); sourceImage = image;
    image.addEventListener("load", () => { if (sourceImage === image) { updateThumbnails(); renderImage(); renderInspector(); if (!snapshot) $("file-info").textContent = `${image.naturalWidth} × ${image.naturalHeight} px · ready to run`; } });
    image.addEventListener("error", () => { if (sourceImage === image) showError("Could not load this image preview. Please choose a valid PNG or JPEG."); });
    image.src = url;
  }
  function display(data) {
    snapshot = data;
    if (settingsJob !== data.id) {
      $("mode").value = data.options.mode; $("budget").value = data.options.max_attempts;
      $("policy").value = data.options.render_policy; $("fragments").checked = data.options.allow_fragments;
      $("translation-policy").value = data.options.translation_policy || "conservative";
      $("replacement-policy").value = data.options.replacement_policy || "all";
      $("quality-reread").checked = Boolean(data.options.quality_reread);
      $("layout-policy").value = data.options.layout_policy || "source-local";
      $("source-localization").checked = Boolean(data.options.source_localization);
      $("visibility-policy").value = data.options.visibility_policy || "replacement-only";
      settingsJob = data.id; settingsSummary();
    }
    if (!pendingFile) $("filename").textContent = `Saved run · ${data.id.slice(0, 8)}`;
    $("file-info").textContent = data.size.join(" × ") + " px · " + (data.status === "running" ? "using the saved upload" : "same upload · retry creates a fresh run");
    $("results").hidden = false; $("empty-state").hidden = true;
    const translations = data.units.filter(unit => unit.status === "preview" && unit.target).length;
    $("status").textContent = data.status === "running" ? `${{starting: "Starting local pipeline", ocr: "Finding and grouping text", translation: "Translating text", rendering: "Cleaning and lettering the page", finalizing: "Finishing outputs"}[data.stage] || "Processing"}… · ${data.elapsed_seconds}s`
      : `${data.status === "partial" || data.status === "complete_with_failures" ? "Finished with partial results" : data.status === "complete" ? "Ready to inspect" : "Run stopped"} · ${data.restored ? "saved run" : data.elapsed_seconds + "s"}`;
    $("status-dot").dataset.state = data.status;
    $("summary").textContent = `${data.detections.length} regions · ${translations} translations${data.display_summary ? ` · ${data.display_summary.displayed_translations} displayed (${data.display_summary.clean_replacements} clean)` : data.rendering_coverage ? ` · ${data.rendering_coverage.rendered} replacements` : ""}`;
    const notice = qualityNotice(data, config?.quality_reread_available);
    $("quality-notice").hidden = !notice;
    $("quality-copy").textContent = notice?.copy || "";
    $("quality-retry").hidden = !notice?.action;
    $("quality-retry").textContent = notice?.action || "";
    const coverage = data.rendering_coverage;
    $("render-note").textContent = data.rendering_skipped ? `No rendered page: ${friendly(data.rendering_skipped)}. Inspect Translation for the available results and blockers.` : coverage
      ? `${coverage.rendered} of ${coverage.saved_units} saved units replaced · ${data.quality?.best_effort_rendered || 0} best-effort.${coverage.partial_output ? " Other source regions remain unchanged." : " Candidate output only."}`
      : data.options.mode === "ocr" ? "OCR-only run: translation and rendering were not requested." : "Translation produces text previews; rendering produces the cleaned and final page images.";
    $("render-note").textContent += localizationNotice(data);
    showError(data.error ? data.error.message || String(data.error) : "");
    loadSource(artifactUrl(data.input));
    if ($("follow").checked) selectedView = progressView(data);
    updateThumbnails(); renderImage(); renderInspector(); renderArtifacts(data); setBusy(data.status === "running");
  }
  async function poll(identity) {
    try {
      const data = await request(`/api/jobs/${identity}`);
      if (identity !== current) return;
      display(data);
      if (data.status === "running") timer = setTimeout(() => poll(identity), 1000);
    } catch (error) {
      if (identity !== current) return;
      showError(`Could not refresh: ${error.message}. Retrying…`); timer = setTimeout(() => poll(identity), 2000);
    }
  }
  function revokeLocal() { if (localURL) { URL.revokeObjectURL(localURL); localURL = null; } }
  function clearArtifacts() {
    for (const id of ["layers", "masks", "artifacts"]) $(id).replaceChildren();
    for (const id of ["ocr-text", "output-directory", "notices"]) $(id).textContent = "";
  }
  function selectJob(identity, keepFile = false) {
    clearTimeout(timer); revokeLocal(); current = identity; snapshot = null; settingsJob = null; sourceImage = null; sourceURL = null; selected = null; outputImages.clear();
    selectedView = "original"; $("follow").checked = true;
    clearArtifacts(); showError(""); $("results").hidden = true;
    $("quality-notice").hidden = true;
    const url = new URL(window.location.href); url.searchParams.set("job", identity); window.history.replaceState(null, "", url);
    if (!keepFile) { pendingFile = null; $("file").value = ""; $("filename").textContent = `Saved run · ${identity.slice(0, 8)}`; $("file-info").textContent = "Original upload preserved on disk"; setBusy(false); }
    poll(identity);
  }
  function chooseFile(file) {
    if (busy || !file) return;
    if (file.size > (config?.max_upload_bytes || 32 * 1024 * 1024)) return showError("Upload limit is 32 MiB.");
    clearTimeout(timer); revokeLocal(); current = null; snapshot = null; settingsJob = null; selected = null; pendingFile = file; outputImages.clear();
    clearArtifacts(); $("follow").checked = true; selectedView = "original";
    $("quality-notice").hidden = true;
    window.history.replaceState(null, "", window.location.pathname);
    $("filename").textContent = file.name; $("file-info").textContent = "Loading image preview…";
    $("results").hidden = false; $("empty-state").hidden = true; $("summary").textContent = ""; $("render-note").textContent = "";
    $("status").textContent = "Image selected. Ready to run."; $("status-dot").dataset.state = "ready"; showError("");
    localURL = URL.createObjectURL(file); loadSource(localURL); updateThumbnails(); renderImage(); renderInspector(); setBusy(false);
  }
  $("run-form").addEventListener("submit", async event => {
    event.preventDefault();
    const retry = current && snapshot && snapshot.status !== "running";
    if ((!pendingFile && !retry) || busy) return;
    const file = pendingFile, parent = current;
    const rendering = $("mode").value === "render";
    const settings = {mode: $("mode").value, max_attempts: Number($("budget").value), allow_fragments: $("fragments").checked, render_policy: $("policy").value, translation_policy: $("translation-policy").value,
      replacement_policy: rendering ? $("replacement-policy").value : "all",
      quality_reread: rendering && $("translation-policy").value === "coverage-first" && !$("quality-reread").disabled && $("quality-reread").checked,
      layout_policy: rendering ? $("layout-policy").value : "source-local",
      source_localization: rendering && $("source-localization").checked};
    settings.visibility_policy = rendering ? $("visibility-policy").value : "replacement-only";
    setBusy(true); showError(""); $("status").textContent = retry ? "Starting a fresh run with the saved image…" : "Uploading original image…";
    try {
      const result = retry
        ? await request(`/api/jobs/${parent}/retry`, {method: "POST", headers: {"Content-Type": "application/json", "X-Tralens-Token": config.token}, body: JSON.stringify(settings)})
        : await request("/api/jobs", {method: "POST", headers: {"Content-Type": "application/octet-stream", "X-Tralens-Token": config.token, "X-Tralens-Options": JSON.stringify(settings)}, body: file});
      const choice = node("option", `${retry ? "Retry " + parent.slice(0, 8) : file.name} · ${result.id.slice(0, 8)}`); choice.value = result.id; $("jobs").append(choice); $("jobs").value = result.id; selectJob(result.id, true);
    } catch (error) { showError(error.message); $("status").textContent = "Run was not started."; setBusy(false); }
  });
  $("quality-retry").addEventListener("click", () => {
    if (busy || !snapshot || !config.render_enabled) return;
    $("mode").value = "render"; $("policy").value = "best-effort"; $("translation-policy").value = "coverage-first";
    $("replacement-policy").value = "confidence-gated"; $("quality-reread").checked = Boolean(config.quality_reread_available);
    $("budget").value = "64"; settingsSummary(); $("run-form").requestSubmit();
  });
  $("file").addEventListener("change", () => chooseFile($("file").files[0]));
  $("jobs").addEventListener("change", () => { if ($("jobs").value) selectJob($("jobs").value); });
  for (const button of $("stages").children) {
    button.addEventListener("click", () => selectView(button.dataset.stage));
    button.addEventListener("keydown", event => {
      const index = VIEW_ORDER.indexOf(button.dataset.stage);
      const next = event.key === "ArrowRight" ? (index + 1) % 5 : event.key === "ArrowLeft" ? (index + 4) % 5 : event.key === "Home" ? 0 : event.key === "End" ? 4 : null;
      if (next !== null) { event.preventDefault(); $("stages").children[next].focus(); selectView(VIEW_ORDER[next]); }
    });
  }
  $("overlay").addEventListener("change", () => { updateThumbnails(); renderImage(); renderInspector(); });
  $("compare").addEventListener("change", renderImage);
  $("zoom").addEventListener("change", () => { $("preview-scroll").dataset.zoom = $("zoom").value; });
  $("follow").addEventListener("change", () => { if ($("follow").checked && snapshot) selectView(progressView(snapshot), false); });
  $("clear-selection").addEventListener("click", () => chooseUnit(null));
  $("show-previous").addEventListener("click", () => selectView(["clean", "final"].includes(selectedView) ? "translation" : "original"));
  $("source").addEventListener("click", event => {
    if (!snapshot || !["ocr", "translation"].includes(selectedView)) return;
    const rect = $("source").getBoundingClientRect(), x = (event.clientX - rect.left) * $("source").width / rect.width, y = (event.clientY - rect.top) * $("source").height / rect.height;
    const context = $("source").getContext("2d"), owners = ownership();
    for (const detection of snapshot.detections) {
      const path = new Path2D(); detection.polygon.forEach(([px, py], i) => i ? path.lineTo(px, py) : path.moveTo(px, py)); path.closePath();
      if (context.isPointInPath(path, x, y)) { const owner = owners.get(JSON.stringify(detection.source_id)); if (owner) chooseUnit(owner.unit.id, true); break; }
    }
  });
  for (const id of ["mode", "budget", "translation-policy", "replacement-policy", "quality-reread", "layout-policy", "source-localization", "visibility-policy"]) $(id).addEventListener("input", settingsSummary);
  for (const eventName of ["dragenter", "dragover"]) document.addEventListener(eventName, event => {
    if (event.dataTransfer?.types.includes("Files")) { event.preventDefault(); if (!busy) document.body.classList.add("dragging"); }
  });
  document.addEventListener("dragleave", event => { if (!event.relatedTarget) document.body.classList.remove("dragging"); });
  document.addEventListener("drop", event => {
    if (!event.dataTransfer?.types.includes("Files")) return;
    event.preventDefault(); document.body.classList.remove("dragging");
    if (!busy && event.dataTransfer.files.length) { $("file").files = event.dataTransfer.files; chooseFile(event.dataTransfer.files[0]); }
  });
  request("/api/config").then(async value => {
    config = value;
    $("configuration").textContent = `${config.device} · ${config.model} · Ollama: ${config.endpoint || "configured local service"} · OCR: ${config.model_root}${config.fonts ? " · Fonts: " + config.fonts : " · No --fonts: rendering disabled"}`;
    $("reread-availability").textContent = config.quality_reread_available ? `Optional crop presentations go to ${config.reread_model} on the configured private Ollama endpoint, with local Manga OCR corroboration. No downloads; agreement is not truth.` : "Not configured: start with --manga-reread-model and an existing snapshot. Artwork checks work without rereads.";
    $("mode").querySelector('[value="render"]').disabled = !config.render_enabled; $("mode").value = config.render_enabled ? "render" : "ocr"; settingsSummary(); setBusy(false);
    const jobs = await request("/api/jobs");
    for (const job of jobs) { const choice = node("option", `Run ${job.id.slice(0, 8)} · ${friendly(job.status)}`); choice.value = job.id; $("jobs").append(choice); }
    if (jobs.length && !pendingFile) { const requested = new URL(window.location.href).searchParams.get("job"); const last = jobs.find(job => job.id === requested) || jobs[jobs.length - 1]; $("jobs").value = last.id; selectJob(last.id); }
  }).catch(error => showError(error.message));
}
