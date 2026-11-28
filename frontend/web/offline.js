// The Android app bundles this module and the reader shell. Only page downloads use the network.
const result = request => new Promise((resolve, reject) => {
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});

export class OfflineLibrary {
  constructor(username) {
    const open = indexedDB.open("tralens-downloads:" + username, 1);
    open.onupgradeneeded = () => {
      open.result.createObjectStore("chapters", {keyPath: "id"});
      open.result.createObjectStore("pages");
    };
    this.database = result(open);
  }
  async run(stores, mode, work) {
    const db = await this.database;
    const tx = db.transaction(stores, mode);
    const done = new Promise((resolve, reject) => {
      tx.oncomplete = resolve;
      tx.onabort = () => reject(tx.error || Error("Storage operation interrupted."));
    });
    try {
      const value = await work(tx.objectStore("chapters"), stores.includes("pages") && tx.objectStore("pages"));
      await done;
      return value;
    } catch (error) {
      try { tx.abort(); } catch {}
      await done.catch(() => {});
      throw error;
    }
  }
  list() { return this.run(["chapters"], "readonly", c => result(c.getAll())); }
  begin(record) {
    return this.run(["chapters"], "readwrite", async c => {
      const existing = await result(c.get(record.id));
      if (existing) return existing;
      const fresh = {...record, generation: crypto.randomUUID(), saved: 0, position: 0};
      c.add(fresh);
      return fresh;
    });
  }
  append(record, index, blob) {
    return this.run(["chapters", "pages"], "readwrite", async (c, p) => {
      const latest = await result(c.get(record.id));
      if (latest?.generation !== record.generation || latest.saved !== index) throw Error("Download changed. Reopen Downloads.");
      p.put(blob, [record.id, index]);
      latest.saved++;
      c.put(latest);
    });
  }
  page(record, index) {
    return this.run(["chapters", "pages"], "readwrite", async (c, p) => {
      const latest = await result(c.get(record.id));
      if (latest?.generation !== record.generation || index < 0 || index >= latest.saved) throw Error("Page unavailable on this device.");
      const blob = await result(p.get([record.id, index]));
      if (!blob) throw Error("Page unavailable on this device.");
      latest.position = index;
      c.put(latest);
      return {blob, saved: latest.saved};
    });
  }
  position(record, index) {
    return this.run(["chapters"], "readwrite", async c => {
      const latest = await result(c.get(record.id));
      if (latest?.generation !== record.generation) throw Error("Download changed. Reopen Downloads.");
      latest.position = index;
      c.put(latest);
    });
  }
  remove(id) {
    return this.run(["chapters", "pages"], "readwrite", (c, p) => {
      p.delete(IDBKeyRange.bound([id, 0], [id, Number.MAX_SAFE_INTEGER]));
      c.delete(id);
    });
  }
}

// A separate dialog keeps offline navigation independent of online jobs, polling and reading position.
export function downloads(username, {escape, notify, direction, unauthorized}) {
  const store = new OfflineLibrary(username);
  const native = window.Capacitor?.isPluginAvailable?.("ReaderDownloads") ? window.Capacitor.Plugins.ReaderDownloads : null;
  const scope = {user: username, origin: location.origin};
  let timer, previousRows = "";
  async function records() {
    const rows = await store.list();
    if (!native || !rows.length) return rows;
    const {jobs} = await native.status(scope);
    return rows.map(row => {
      const job = jobs[row.generation];
      return job ? {...row, localSaved: row.saved, saved: Math.max(row.saved, job.saved), state: job.state, error: job.error} : row;
    });
  }
  const dialog = document.createElement("dialog");
  dialog.className = "offline-library";
  dialog.setAttribute("aria-label", "Downloads");
  document.body.append(dialog);
  let controller, saving, active, objectURL, serial = 0, disposed = false, feedback = "";
  const button = (action, label) => '<button class="btn" data-local="' + action + '">' + label + '</button>';
  function releaseImage() {
    if (objectURL?.startsWith("blob:")) URL.revokeObjectURL(objectURL);
    objectURL = null;
  }
  function errorMessage(err) {
    return err.name === "QuotaExceededError" ? "Device storage is full. Remove a download and resume." : err.message;
  }
  async function list() {
    const version = ++serial;
    active = null;
    const rows = await records();
    if (version !== serial || disposed) return;
    previousRows = JSON.stringify(rows);
    releaseImage();
    if (!dialog.open) dialog.showModal();
    dialog.innerHTML = '<header class="line"><h1>Downloads</h1>' + button("close", "Close") + '</header>' +
      (feedback ? '<p role="alert" class="error">' + escape(feedback) + '</p>' : "") +
      (rows.length ? rows.map(r => {
        const running = r.id === saving || r.state === "queued";
        return '<section class="panel" data-chapter-id="' + escape(r.id) + '"><h2>' + escape(r.book) + '</h2><p>' +
          escape(r.title) + '</p>' + (r.originals ? '<p class="meta">' + r.originals + ' ' +
            (r.originals === 1 ? 'page uses' : 'pages use') + ' original artwork in this download.</p>' : '') +
          (r.error ? '<p class="error">' + escape(r.error) + '</p>' : '') +
          '<p class="meta" role="status">' + r.saved + ' of ' + r.urls.length + ' pages saved</p><div class="actions">' +
          ('<button class="btn primary" data-local="read"' + (r.saved ? '' : ' hidden') + '>Read offline</button>') + (running ? button("pause", "Pause") :
            (r.saved < r.urls.length ? button("resume", "Resume download") : "") + button("remove", "Remove download")) +
          '</div></section>';
      }).join("") : '<p>No chapters saved. Open a chapter in your library and choose Save offline.</p>');
    schedule(rows);
  }
  function updateTotal(total) {
    if (!active || total === active.saved) return;
    active.saved = total;
    const slider = dialog.querySelector('input[type="range"]');
    slider.max = String(total);
    slider.disabled = total <= 1;
    dialog.querySelector("[data-total]").textContent = total;
    dialog.querySelector('[data-local="next"]').disabled = active.position + 1 >= total;
  }
  function schedule(rows) {
    clearTimeout(timer);
    if (native && !document.hidden && dialog.open && rows.some(r => r.saved < r.urls.length))
      timer = setTimeout(() => sync().catch(err => notify(errorMessage(err))), 1200);
  }
  async function sync() {
    if (!native || disposed || !dialog.open || document.hidden) return;
    const version = serial;
    const rows = await records();
    if (version !== serial || disposed || !dialog.open) return;
    if (active) {
      const row = rows.find(r => r.generation === active.generation);
      if (row) updateTotal(row.saved);
    } else if (JSON.stringify(rows) !== previousRows) return list();
    schedule(rows);
  }
  function visibility() {
    if (document.hidden) clearTimeout(timer);
    else sync().catch(err => notify(errorMessage(err)));
  }
  document.addEventListener("visibilitychange", visibility);
  async function read(id, index) {
    const version = ++serial;
    const row = active?.id === id ? active : (await records()).find(r => r.id === id);
    if (!row || version !== serial || disposed) return;
    const i = index ?? row.position;
    let blob, path;
    if (native && row.localSaved !== undefined && i >= row.localSaved) {
      ({path} = await native.page({...scope, generation: row.generation, index: i}));
      await store.position(row, i);
    } else {
      ({blob} = await store.page(row, i));
    }
    const saved = row.saved;
    if (version !== serial || disposed || !dialog.open) return;
    releaseImage();
    objectURL = path ? window.Capacitor.convertFileSrc(path) : URL.createObjectURL(blob);
    active = {...row, saved, position: i};
    const focused = dialog.querySelector(":focus");
    dialog.innerHTML = '<div class="offline-reader"><header class="line">' + button("list", "Downloads") +
      '<b>' + escape(row.title) + '</b></header><div class="offline-canvas"><img alt="Page ' + (i+1) +
      '" src="' + objectURL + '"></div><footer class="line">' +
      '<button class="btn" data-local="prev"' + (i ? '' : ' disabled') + '>Previous</button>' +
      '<label class="offline-progress"><output>Page ' + (i+1) + ' of <span data-total>' + saved +
      '</span></output><input type="range" aria-label="Saved page" min="1" max="' + saved + '" value="' + (i+1) +
      '" dir="' + (direction() === "rtl" ? "rtl" : "ltr") + '"' + (saved === 1 ? ' disabled' : '') +
      '></label><button class="btn" data-local="next"' +
      (i+1 < saved ? '' : ' disabled') + '>Next</button></footer></div>';
    schedule([active]);
    const control = focused?.dataset.local;
    if (focused?.type === "range") dialog.querySelector('input[type="range"]').focus({preventScroll:true});
    else if (control === "prev" || control === "next") {
      const target = dialog.querySelector('[data-local="' + control + '"]');
      (target.disabled ? dialog.querySelector('input[type="range"]') : target).focus({preventScroll:true});
    }
  }
  async function save(record) {
    if (native) {
      navigator.storage?.persist?.().catch(() => {});
      feedback = "";
      const row = await store.begin(record);
      if (row.saved < row.urls.length) {
        await native.start({...scope, generation: row.generation, title: row.title, urls: row.urls, offset: row.saved});
      }
      if (!disposed) await list();
      return;
    }
    if (controller) { await list(); return; }
    const abort = new AbortController();
    feedback = "";
    controller = abort;
    saving = record.id;
    try {
      navigator.storage?.persist?.().catch(() => {});
      const row = await store.begin(record);
      await list();
      for (let i = row.saved; i < row.urls.length; i++) {
        const response = await fetch(row.urls[i], {signal: abort.signal, redirect: "error"});
        if (response.status === 401) { unauthorized(); return; }
        if (!response.ok || !response.headers.get("Content-Type")?.startsWith("image/")) throw Error("Page unavailable. Reconnect and resume the download.");
        const blob = await response.blob();
        if (!blob.size) throw Error("Empty page. Resume to try again.");
        abort.signal.throwIfAborted();
        await store.append(row, i, blob);
        const status = dialog.querySelector('[data-chapter-id="' + row.id + '"] [role=status]');
        if (status) {
          status.textContent = (i+1) + " of " + row.urls.length + " pages saved";
          status.closest("section").querySelector('[data-local="read"]').hidden = false;
        }
        if (active?.generation === row.generation) updateTotal(i + 1);
      }
      if (!disposed) notify("Chapter saved for offline reading.");
    } catch (err) {
      if (!abort.signal.aborted && !disposed) {
        feedback = err instanceof TypeError ? "Download interrupted. Reconnect and tap Resume." : errorMessage(err);
        if (!dialog.open) notify(feedback);
      }
    } finally {
      controller = saving = null;
      if (!disposed && dialog.open && !active) await list();
    }
  }
  function turn(step) {
    const next = active?.position + step;
    if (active && next >= 0 && next < active.saved) read(active.id, next).catch(e => notify(errorMessage(e)));
  }
  function back() {
    if (!dialog.open) return false;
    if (active) list().catch(e => notify(errorMessage(e)));
    else dialog.close();
    return true;
  }
  dialog.oncancel = e => { e.preventDefault(); back(); };
  dialog.onclose = () => { if (location.hash === "#downloads") history.replaceState(null, "", "#library"); clearTimeout(timer); ++serial; active = null; releaseImage(); };
  dialog.onchange = e => { if (active && e.target.matches('input[type="range"]')) read(active.id, Number(e.target.value) - 1).catch(err => notify(errorMessage(err))); };
  dialog.onkeydown = e => {
    if (!active || e.target.matches('input[type="range"]')) return;
    if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
      e.preventDefault(); e.stopPropagation();
      turn((e.key === "ArrowLeft") === (direction() === "rtl") ? 1 : -1);
    }
  };
  let swipe;
  dialog.addEventListener("touchstart", e => {
    swipe = e.touches.length === 1 && e.target.closest(".offline-canvas") ? e.touches[0] : null;
  }, {passive: true});
  dialog.addEventListener("touchend", e => {
    if (!swipe) return;
    const dx = e.changedTouches[0].clientX - swipe.clientX;
    const dy = e.changedTouches[0].clientY - swipe.clientY;
    swipe = null;
    if (Math.abs(dx) > 65 && Math.abs(dx) > 1.6*Math.abs(dy) && (window.visualViewport?.scale ?? 1) <= 1)
      turn((dx > 0) === (direction() === "rtl") ? 1 : -1);
  }, {passive: true});
  dialog.onclick = async e => {
    const cmd = e.target.closest("[data-local]")?.dataset.local;
    if (!cmd) return;
    const id = e.target.closest("[data-chapter-id]")?.dataset.chapterId;
    try {
      if (cmd === "close") dialog.close();
      if (cmd === "list") await list();
      if (cmd === "read") await read(id);
      if (cmd === "prev" || cmd === "next") turn(cmd === "next" ? 1 : -1);
      if (cmd === "pause" || cmd === "remove") {
        const row = (await store.list()).find(r => r.id === id);
        if (native && row) await native.change({...scope, generation: row.generation, remove: cmd === "remove"});
        if (cmd === "pause") controller?.abort();
        if (cmd === "remove") await store.remove(id);
        feedback = "";
        await list();
      }
      if (cmd === "resume") { const row = (await records()).find(r => r.id === id); if (row) await save(row); }
    } catch (err) { notify(errorMessage(err)); }
  };
  return {list, save, back, isOpen: () => dialog.open,
    hasSaved: async () => (await store.list()).length > 0,
    dispose() { document.removeEventListener("visibilitychange", visibility); clearTimeout(timer); disposed = true; ++serial; controller?.abort(); dialog.remove(); releaseImage(); store.database.then(db => db.close(), () => {}); }};
}
