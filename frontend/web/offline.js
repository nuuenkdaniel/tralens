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
  remove(id) {
    return this.run(["chapters", "pages"], "readwrite", (c, p) => {
      p.delete(IDBKeyRange.bound([id, 0], [id, Number.MAX_SAFE_INTEGER]));
      c.delete(id);
    });
  }
}

// A separate dialog keeps offline navigation independent of online jobs, polling and reading position.
export function downloads(username, {escape, notify, direction, unauthorized, progressChanged}) {
  const store = new OfflineLibrary(username);
  const dialog = document.createElement("dialog");
  dialog.className = "offline-library";
  dialog.setAttribute("aria-label", "Downloads");
  document.body.append(dialog);
  let controller, saving, active, objectURL, serial = 0, disposed = false, feedback = "";
  const progress = new Map();
  function remember(id, saved, total) {
    if (disposed) return;
    progress.set(id, {saved, total, running: saving === id});
    progressChanged(id);
  }
  const restored = store.list().then(rows => {
    for (const row of rows) remember(row.id, row.saved, row.urls.length);
  }).catch(() => {});
  const button = (action, label) => '<button class="btn" data-local="' + action + '">' + label + '</button>';
  function releaseImage() {
    if (objectURL) URL.revokeObjectURL(objectURL);
    objectURL = null;
  }
  function errorMessage(err) {
    return err.name === "QuotaExceededError" ? "Device storage is full. Remove a download and resume." : err.message;
  }
  async function list() {
    const version = ++serial;
    active = null;
    const rows = await store.list();
    if (version !== serial || disposed) return;
    releaseImage();
    if (!dialog.open) dialog.showModal();
    dialog.innerHTML = '<header class="line"><h1>Downloads</h1>' + button("close", "Close") + '</header>' +
      (feedback ? '<p role="alert" class="error">' + escape(feedback) + '</p>' : "") +
      (rows.length ? rows.map(r => {
        const running = r.id === saving;
        return '<section class="panel" data-chapter-id="' + escape(r.id) + '"><h2>' + escape(r.book) + '</h2><p>' +
          escape(r.title) + '</p>' + (r.originals ? '<p class="meta">' + r.originals + ' ' +
            (r.originals === 1 ? 'page uses' : 'pages use') + ' original artwork in this download.</p>' : '') +
          '<p class="meta" role="status">' + r.saved + ' of ' + r.urls.length + ' pages saved</p><div class="actions">' +
          ('<button class="btn primary" data-local="read"' + (r.saved ? '' : ' hidden') + '>Read offline</button>') + (running ? button("pause", "Pause") :
            (r.saved < r.urls.length ? button("resume", "Resume download") : "") + button("remove", "Remove download")) +
          '</div></section>';
      }).join("") : '<p>No chapters saved. Open a chapter in your library and choose Save offline.</p>');
  }
  async function read(id, index) {
    const version = ++serial;
    const row = active?.id === id ? active : (await store.list()).find(r => r.id === id);
    if (!row || version !== serial || disposed) return;
    const i = index ?? row.position;
    const {blob, saved} = await store.page(row, i);
    if (version !== serial || disposed || !dialog.open) return;
    releaseImage();
    objectURL = URL.createObjectURL(blob);
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
    const control = focused?.dataset.local;
    if (focused?.type === "range") dialog.querySelector('input[type="range"]').focus({preventScroll:true});
    else if (control === "prev" || control === "next") {
      const target = dialog.querySelector('[data-local="' + control + '"]');
      (target.disabled ? dialog.querySelector('input[type="range"]') : target).focus({preventScroll:true});
    }
  }
  async function save(record) {
    await restored;
    if (disposed) return;
    const previous = progress.get(record.id);
    if (controller || (previous && previous.saved === previous.total)) { await list(); return; }
    const abort = new AbortController();
    feedback = "";
    controller = abort;
    saving = record.id;
    remember(record.id, previous?.saved ?? 0, previous?.total ?? record.urls.length);
    try {
      navigator.storage?.persist?.().catch(() => {});
      const row = await store.begin(record);
      remember(row.id, row.saved, row.urls.length);
      if (dialog.open && !active) await list();
      for (let i = row.saved; i < row.urls.length; i++) {
        const response = await fetch(row.urls[i], {signal: abort.signal, redirect: "error"});
        if (response.status === 401) { unauthorized(); return; }
        if (!response.ok || !response.headers.get("Content-Type")?.startsWith("image/")) throw Error("Page unavailable. Reconnect and resume the download.");
        const blob = await response.blob();
        if (!blob.size) throw Error("Empty page. Resume to try again.");
        abort.signal.throwIfAborted();
        await store.append(row, i, blob);
        remember(row.id, i + 1, row.urls.length);
        const status = dialog.querySelector('[data-chapter-id="' + row.id + '"] [role=status]');
        if (status) {
          status.textContent = (i+1) + " of " + row.urls.length + " pages saved";
          status.closest("section").querySelector('[data-local="read"]').hidden = false;
        }
        if (active?.generation === row.generation && active.saved <= i) {
          active.saved = i+1;
          const slider = dialog.querySelector('input[type="range"]');
          slider.max = String(active.saved);
          slider.disabled = false;
          dialog.querySelector("[data-total]").textContent = active.saved;
          dialog.querySelector('[data-local="next"]').disabled = false;
        }
      }
      if (!disposed) notify("Chapter saved for offline reading.");
    } catch (err) {
      if (!abort.signal.aborted && !disposed) {
        feedback = err instanceof TypeError ? "Download interrupted. Reconnect and tap Resume." : errorMessage(err);
        if (!dialog.open) notify(feedback);
      }
    } finally {
      controller = saving = null;
      const last = progress.get(record.id);
      remember(record.id, last.saved, last.total);
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
  dialog.onclose = () => { ++serial; active = null; releaseImage(); };
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
      if (cmd === "pause") controller?.abort();
      if (cmd === "remove") { await store.remove(id); progress.delete(id); if (!disposed) progressChanged(id); feedback = ""; await list(); }
      if (cmd === "resume") { const row = (await store.list()).find(r => r.id === id); if (row) await save(row); }
    } catch (err) { notify(errorMessage(err)); }
  };
  return {list, save, back, progress: id => progress.get(id), isOpen: () => dialog.open,
    hasSaved: async () => (await store.list()).length > 0,
    dispose() { disposed = true; ++serial; controller?.abort(); dialog.remove(); releaseImage(); store.database.then(db => db.close(), () => {}); }};
}
