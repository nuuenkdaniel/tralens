const $ = (selector) => document.querySelector(selector);
const root = $("#root"),
  sheet = $("#sheet");
const defaults = {
  translator: "local",
  paint: "auto",
  protection: "all",
  direction: "rtl",
  layout: "single",
  fit: "screen",
  background: "dark",
  theme: "system",
  comparison: "swipe",
  sort: "recent",
};
let prefs = { ...defaults };
try {
  Object.assign(
    prefs,
    JSON.parse(localStorage.getItem("tralens-reader") || "{}"),
  );
} catch {}
let config,
  catalog = { books: [], bytes: 0 },
  query = "",
  view = "translated",
  draft,
  noticeTimer,
  hideTimer;
let pendingPosition = Promise.resolve(),
  focusReturn,
  dismissed = "",
  pendingImport = null;
const esc = (text) =>
  String(text ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const icon = (name) =>
  '<svg class="icon" aria-hidden="true"><use href="#i-' +
  name +
  '"></use></svg>';
const btn = (action, label, cls = "", symbol = "") =>
  '<button type="button" class="' +
  cls +
  '" data-action="' +
  action +
  '">' +
  (symbol ? icon(symbol) : "") +
  esc(label) +
  "</button>";
const ib = (action, label, symbol) =>
  '<button class="icon-btn" type="button" data-action="' +
  action +
  '" aria-label="' +
  esc(label) +
  '" title="' +
  esc(label) +
  '">' +
  icon(symbol) +
  "</button>";
const pagesOf = (book) => book.chapters.flatMap((c) => c.pages);
const url = (p, which, suffix = "") =>
  "/api/library/pages/" +
  p.id +
  "/" +
  which + suffix +
  (which === "translated" ? "?v=" + p.updated : "");
const best = (p) => url(p, p.ready ? "translated" : "original");
const thumbnail = (p) => url(p, p.ready ? "translated" : "original", "/thumbnail");
const readURL = (b, p) => "#read/" + b.id + "/" + p.id;
const lastPage = (b) =>
  pagesOf(b).find((p) => p.id === b.position) || pagesOf(b)[0];
const route = () => location.hash.slice(1).split("/");
function current() {
  const [screen, id, pid] = route(),
    book = catalog.books.find((b) => b.id === id);
  const chapter = book?.chapters.find((c) => c.pages.some((p) => p.id === pid));
  return {
    screen,
    book,
    chapter,
    page: chapter?.pages.find((p) => p.id === pid),
  };
}
function save() {
  localStorage.setItem(config?.username ? "tralens-reader:" + config.username : "tralens-reader", JSON.stringify(prefs));
  document.documentElement.dataset.theme = prefs.theme;
}
document.documentElement.dataset.theme = prefs.theme;
function toast(message) {
  $("#notice").textContent = message;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => ($("#notice").textContent = ""), 4500);
}
async function api(path, body, reconnect = true, send = fetch) {
  const opts =
    body === undefined
      ? {}
      : { method: "POST", headers: { "X-Tralens-Token": config.token }, body };
  if (body !== undefined && !(body instanceof FormData)) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  if (path === "config" || path.startsWith("library/imports/")) opts.signal = AbortSignal.timeout(6000);
  let res;
  try {
    res = await send("/api/" + path, opts);
  } catch (err) {
    if (err instanceof TypeError) throw Error("Couldn’t reach your server. Check your connection.");
    throw err;
  }
  if (res.status === 401) {
    config = null;
    showAccountForm();
    throw Error("Sign in to your Tralens server.");
  }
  const data = res.ok ? await res.json() : await res.json().catch(() => ({}));
  if (
    res.status === 403 &&
    reconnect &&
    body !== undefined &&
    /token|reload/i.test(data.error || "")
  ) {
    const fresh = await api("config");
    if (fresh.username !== config?.username) {
      location.reload();
      throw Error("The signed-in account changed.");
    }
    config = fresh;
    return api(path, body, false, send);
  }
  if (!res.ok) throw Error(data.error || "Server request failed (HTTP " + res.status + "). Try again.");
  return data;
}
let pollTimer, refreshId = 0;
function schedulePoll() {
  clearTimeout(pollTimer);
  if (document.hidden || !config) return;
  const busy = catalog.books.some(b => b.chapters.some(c =>
    c.pages.some(p => ["queued", "running", "waiting"].includes(p.status))));
  pollTimer = setTimeout(poll, busy ? 2500 : 60000);
}
async function poll() {
  if (!config || document.hidden) return schedulePoll();
  try { await refresh(); }
  catch (err) { toast(err.message); schedulePoll(); }
}
async function refresh(force = false) {
  const id = ++refreshId;
  const data = await api("library");
  if (id !== refreshId || !config) return;
  const before = current();
  const previous = catalog;
  catalog = data;
  schedulePoll();
  const after = current();
  if (!force && before.page && after.page && $(".reader") &&
      before.book.title === after.book.title && before.chapter.title === after.chapter.title &&
      before.book.chapters.at(-1).id === after.book.chapters.at(-1).id &&
      before.chapter.pages.length === after.chapter.pages.length) {
    const continuous = $(".continuous");
    const start = after.chapter.pages.indexOf(after.page);
    const count = prefs.layout === "spread" && view !== "compare" ? 2 : 1;
    const visible = continuous ? after.chapter.pages : after.chapter.pages.slice(start, start + count);
    const offset = continuous ? 0 : start;
    for (const [i, page] of visible.entries()) {
      const old = before.chapter.pages[offset + i];
      if (JSON.stringify(old) === JSON.stringify(page)) continue;
      if (!continuous || old.id !== page.id) return render();
      continuous.children[i].outerHTML = pageImage(page, view);
    }
    if (continuous && JSON.stringify(before.page) !== JSON.stringify(after.page)) {
      $("#page-status").innerHTML = pageBanner(after.page);
    }
    return;
  }
  if (force || JSON.stringify(previous) !== JSON.stringify(data)) render();
}
function settings() {
  const protect = prefs.protection === "protect";
  return {
    mode: "render",
    translation_model: prefs.translator,
    reconstruction_policy: protect ? "classical" : prefs.paint,
    replacement_policy: protect ? "confidence-gated" : "all",
    removal_policy: protect ? "preserve" : "aggressive",
    visibility_policy: protect ? "replacement-only" : "always-show",
    luna_mode: "reviewed",
  };
}
function privacy() {
  return prefs.translator === "local"
    ? "Pages stay on your server. Text is sent to your configured Ollama service."
    : "Page images and detected text are sent to OpenAI through the account connected to your server.";
}
function header(back) {
  return (
    '<header class="hdr">' +
    (back
      ? '<a href="#library" class="btn">' + icon("chev-l") + "Library</a>"
      : '<a href="#library" class="brand"><span class="mark">文<b>A</b></span>Tralens</a>') +
    '<span class="grow"></span>' +
    (!back ? btn("add", "Add manga", "primary wide", "plus") : "") +
    '<a class="btn icon-btn" href="#settings" aria-label="Settings">' +
    icon("gear") +
    "</a>" +
    ib("theme", "Switch theme", "contrast") +
    "</header>"
  );
}
function statusText(pages) {
  const working = pages.filter((p) =>
      ["queued", "running"].includes(p.status),
    ).length,
    failed = pages.filter((p) => p.status === "failed").length;
  return [
    working
      ? "Translating · " +
        (pages.length - working) +
        " of " +
        pages.length +
        " processed"
      : "",
    failed ? failed + " failed" : "",
    pages.some(p => p.status === "waiting") ? "Waiting for local translator" : "",
  ]
    .filter(Boolean)
    .join(" · ");
}
function card(b) {
  const pages = pagesOf(b),
    p = lastPage(b),
    idx = pages.indexOf(p) + 1;
  return (
    '<a class="book" href="#book/' +
    b.id +
    '"><img loading="lazy" src="' +
    thumbnail(pages[0]) +
    '" alt=""><b>' +
    esc(b.title) +
    "</b><small>" +
    b.chapters.length +
    " ch. · " +
    (b.position ? "Page " + idx + " of " : "") +
    pages.length +
    (b.position ? "" : " pages") +
    "</small>" +
    (b.position
      ? '<progress class="bar" value="' +
        idx +
        '" max="' +
        pages.length +
        '"></progress>'
      : "") +
    "<small>" +
    esc(statusText(pages)) +
    "</small></a>"
  );
}
function shelf() {
  const books = catalog.books.filter((b) =>
    b.title.toLocaleLowerCase().includes(query.toLocaleLowerCase()),
  );
  books.sort((a, b) =>
    prefs.sort === "title"
      ? a.title.localeCompare(b.title)
      : prefs.sort === "added"
        ? b.added - a.added
        : (b.last_read || b.added) - (a.last_read || a.added),
  );
  return books.map(card).join("") || '<p class="meta">No matching manga.</p>';
}
function library() {
  if (!catalog.books.length)
    return (
      header(false) +
      '<main class="empty"><h1>Your library is <em>empty</em></h1><p class="meta">Add Japanese manga pages to read them in English, side by side with the originals.</p>' +
      btn("add", "Add manga", "primary", "plus") +
      '<p class="hint">PNG, JPEG, WebP, PDF, CBZ or ZIP · Drop files here to get started</p></main>'
    );
  const recent = [...catalog.books]
    .filter((b) => b.position)
    .sort((a, b) => b.last_read - a.last_read)[0];
  const hero = recent
    ? '<section class="hero"><img alt="" src="' +
      thumbnail(lastPage(recent)) +
      '"><div><small>Continue reading</small><h2>' +
      esc(recent.title) +
      '</h2><p class="meta">' +
      esc(
        recent.chapters.find((c) => c.pages.includes(lastPage(recent))).title,
      ) +
      '</p><a class="btn primary" href="' +
      readURL(recent, lastPage(recent)) +
      '">' +
      icon("play") +
      "Resume</a></div></section>"
    : "";
  return (
    header(false) +
    '<main class="wrap"><h1>Your library</h1>' +
    hero +
    '<div class="tools"><input id="search" type="search" aria-label="Search manga" placeholder="Search your library" value="' +
    esc(query) +
    '">' +
    select(
      "sort",
      [
        ["recent", "Recently read"],
        ["added", "Recently added"],
        ["title", "Title"],
      ],
      "Sort library",
    ) +
    '</div><div class="grid" id="shelf">' +
    shelf() +
    "</div></main>" +
    btn("add", "Add manga", "primary fab", "plus")
  );
}
function select(key, choices, label) {
  return (
    '<select aria-label="' +
    esc(label) +
    '" data-pref="' +
    key +
    '">' +
    choices
      .map(
        ([v, t, off]) =>
          '<option value="' +
          v +
          '" ' +
          (prefs[key] === v ? "selected " : "") +
          (off ? "disabled" : "") +
          ">" +
          esc(t) +
          (off ? " · unavailable" : "") +
          "</option>",
      )
      .join("") +
    "</select>"
  );
}
const readingFields = [
  [
    "direction",
    "Reading direction",
    [
      ["rtl", "Right to left"],
      ["ltr", "Left to right"],
    ],
  ],
  [
    "layout",
    "Page layout",
    [
      ["single", "Single page"],
      ["spread", "Two-page spread"],
      ["continuous", "Continuous scroll"],
    ],
  ],
  [
    "fit",
    "Page size",
    [
      ["screen", "Fit screen"],
      ["width", "Fit width"],
    ],
  ],
  [
    "background",
    "Reader background",
    [
      ["dark", "Dark"],
      ["paper", "Paper"],
    ],
  ],
];
const field = ([k, l, choices]) =>
  '<label class="field">' + l + select(k, choices, l) + "</label>";

function preferences() {
  const translators = [
    ["local", "Local · " + config.model],
    ["gpt-6-luna", "GPT-6 Luna"],
    ["gpt-6.1-terra", "GPT-6.1 Terra"],
    ["gpt-6.2-sol", "GPT-6.2 Sol"],
  ].filter(([id]) => id === "local" || !config.chatgpt_models || config.chatgpt_models.includes(id));
  return (
    header(true) +
    '<main class="wrap settings"><h1>Settings</h1>' +
    '<section class="panel fields"><h2>Server</h2><p>' + esc(location.origin) + '</p>' + btn("configure-server", "Change server", "", "server") + '</section>' +
    '<section class="panel fields"><h2>Account</h2><p>Signed in as <b>' + esc(config.username) + '</b></p>' + btn("owner-logout", "Sign out") + "</section>" +
    '<section class="panel fields"><h2>Translation</h2>' +
    field(["translator", "Translation model", translators]) +
    '<p class="hint">' +
    esc(privacy()) +
    "</p>" +
    (config.chatgpt_connected ? btn("disconnect-chatgpt", "Disconnect ChatGPT") : "") +
    field([
      "protection",
      "When text is hard to remove",
      [
        ["all", "Replace it anyway"],
        ["protect", "Keep the artwork"],
      ],
    ]) +
    field([
      "paint",
      "Inpainting model",
      [
        ["auto", "Auto · per region"],
        ["classical", "Classical"],
        ["lama", "LaMa", !config.reconstruction_available],
        ["sdxl", "SDXL · GPU", !config.sdxl_available],
      ],
    ]) +
    (prefs.protection === "protect"
      ? '<p class="hint">Keep the artwork uses classical cleanup. Your model choice applies when replacing all text.</p>'
      : "") +
    "</section>" +
    '<section class="panel fields"><h2>Reading</h2>' +
    readingFields.map(field).join("") +
    field([
      "theme",
      "Appearance",
      [
        ["system", "System"],
        ["light", "Light"],
        ["dark", "Dark"],
      ],
    ]) +
    "</section>" +
    '<section class="panel"><h2>Storage</h2><p>Your library is saved on this server · ' +
    (catalog.bytes / 1048576).toFixed(1) +
    ' MiB of page images.</p><p class="hint">Reading preferences are saved on this device.</p></section></main>'
  );
}
function detail(b) {
  const pages = pagesOf(b);
  return (
    header(true) +
    '<main class="wrap detail"><img class="cover" src="' +
    thumbnail(pages[0]) +
    '" alt="' +
    esc(b.title) +
    ' cover"><div><h1>' +
    esc(b.title) +
    '</h1><p class="meta">' +
    b.chapters.length +
    " chapters · " +
    pages.length +
    ' pages</p><div class="actions"><a class="btn primary" href="' +
    readURL(b, lastPage(b)) +
    '">' +
    icon("play") +
    (b.position ? "Continue reading" : "Start reading") +
    "</a>" +
    btn("download-book", "Download", "", "download") +
    ib("book-menu", "Manga actions", "more") +
    '</div><div class="line"><h2>Chapters</h2>' +
    btn("add-chapter", "Add chapter", "", "plus") +
    "</div>" +
    b.chapters
      .map(
        (c, i) =>
          '<a class="chapter btn" href="' +
          readURL(b, c.pages.find((p) => p.id === c.position) || c.pages[0]) +
          '"><span>' +
          esc(c.title) +
          "</span><small>" +
          c.pages.length +
          " pages " +
          esc(
            [
              statusText(c.pages),
              c.position
                ? c.pages.at(-1).id === c.position
                  ? "Read"
                  : "Page " +
                    (c.pages.findIndex((p) => p.id === c.position) + 1)
                : "",
            ]
              .filter(Boolean)
              .join(" · "),
          ) +
          "</small></a><details><summary>Pages in " +
          esc(c.title) +
          '</summary><div class="pages" data-chapter="' + c.id + '"></div></details>',
      )
      .join("") +
    "</div></main>"
  );
}
root.addEventListener("toggle", (event) => {
  if (!event.target.matches("details[open]")) return;
  const list = event.target.querySelector(".pages[data-chapter]");
  if (!list || list.childElementCount) return;
  const { book } = current();
  const chapter = book?.chapters.find(c => c.id === list.dataset.chapter);
  if (chapter) list.innerHTML = chapter.pages.map((p, i) =>
    '<a href="' + readURL(book, p) + '"><img loading="lazy" src="' + thumbnail(p) +
    '" alt="Page ' + (i + 1) + '">' + (i + 1) + '</a>').join("");
}, true);
function pageImage(p, mode) {
  const columns = prefs.layout === "spread" || (view === "compare" && prefs.comparison === "side") ? 2 : 1;
  const width = Math.min(innerWidth / columns, prefs.fit === "width" ? Infinity : innerHeight * p.width / p.height, p.width);
  const reading = mode !== "original" && p.has_text;
  const tapToRead = reading && view !== "compare" && (p.reading_width < 0 || width < p.reading_width);
  const translated = mode !== "original" && p.ready;
  return (
    '<div class="imagebox" data-page="' +
    p.id +
    '"' + (tapToRead ? ' data-reading-fallback' : '') + '><img class="page-image" ' +
    (p.width && p.height
      ? 'width="' + p.width + '" height="' + p.height + '" '
      : "") +
    'src="' +
    (translated ? best(p) : url(p, "original")) +
    '" alt="' +
    esc(p.name) +
    " · " +
    (translated ? "translated" : "original") +
    (prefs.layout === "continuous" && view !== "compare" ? '" loading="lazy">' : '">') +
    (mode !== "original" && !p.ready
      ? '<small class="badge">Original</small>'
      : "") +
    "</div>"
  );
}
function pageBanner(p) {
  const pending = ["queued", "running"].includes(p.status),
    error = p.error || (p.status === "skipped" ? "Translation skipped." : null);
  return view !== "original" &&
    (pending || error) &&
    dismissed !== p.id + ":" + p.status
      ? '<div class="banner">' +
        esc(
          pending
            ? p.status === "queued"
              ? "Waiting to translate…"
              : "Translating this page…"
            : error,
        ) +
        (!pending && p.status !== "waiting" ? btn("retry-page", "Try again") : "") +
        ib("dismiss", "Dismiss", "x") +
        "</div>"
      : "";
}
function reader(b, c, p) {
  const i = c.pages.indexOf(p),
    rtl = prefs.direction === "rtl",
    pair = c.pages.slice(i, i + 2);
  const compare = view === "compare" && p.ready;
  let content = pageImage(p, view);
  if (compare && prefs.comparison === "side")
    content =
      '<div class="sidepair">' +
      pageImage(p, "original") +
      pageImage(p, "translated") +
      "</div>";
  else if (compare)
    content =
      '<div class="wipe"><img class="page-image" src="' +
      best(p) +
      '" alt="Translated"><img class="page-image overlay" src="' +
      url(p, "original") +
      '" alt="Original"><input id="wipe" type="range" value="50" aria-label="Original and translated divider"></div>';
  else if (prefs.layout === "spread" && view !== "compare")
    content =
      '<div class="spread">' +
      (rtl ? pair.reverse() : pair).map((x) => pageImage(x, view)).join("") +
      "</div>";
  else if (prefs.layout === "continuous" && view !== "compare")
    content = c.pages.map((x) => pageImage(x, view)).join("");
  return (
    '<main class="reader ' +
    (prefs.background === "paper" ? "paper " : "") +
    (prefs.fit === "width" ? "width" : "") +
    '"><div class="canvas ' +
    (prefs.layout === "continuous" && view !== "compare" ? "continuous" : "") +
    '">' +
    content +
    "</div>" +
    '<div id="page-status">' + pageBanner(p) + "</div>" +
    '<header><a class="btn icon-btn" href="#book/' +
    b.id +
    '" aria-label="Back to manga">' +
    icon("chev-l") +
    '</a><div class="title"><b>' +
    esc(b.title) +
    "</b><small>" +
    esc(c.title) +
    '</small></div><span class="grow"></span><div class="seg">' +
    ["translated", "original", "compare"]
      .map(
        (v) =>
          '<button data-action="' +
          v +
          '" class="' +
          (v === view ? "active" : "") +
          '" aria-pressed="' +
          (v === view) +
          '">' +
          v[0].toUpperCase() +
          v.slice(1) +
          "</button>",
      )
      .join("") +
    '</div><span class="grow"></span>' +
    (compare
      ? ib(
          "toggle-comparison",
          prefs.comparison === "side"
            ? "Switch to swipe comparison"
            : "Switch to side-by-side comparison",
          prefs.comparison === "side" ? "split" : "columns",
        )
      : "") +
    ib("reader-prefs", "Reader settings", "gear") +
    ib("page-menu", "Page actions", "more") +
    "</header>" +
    '<button class="edge left" data-action="' +
    (rtl ? "next" : "prev") +
    '" aria-label="' +
    (rtl ? "Next" : "Previous") +
    ' page">' +
    icon("chev-l") +
    '</button><button class="edge right" data-action="' +
    (rtl ? "prev" : "next") +
    '" aria-label="' +
    (rtl ? "Previous" : "Next") +
    ' page">' +
    icon("chev-r") +
    "</button>" +
    "<footer>" +
    ib(
      rtl ? "next-ch" : "prev-ch",
      rtl ? "Next chapter" : "Previous chapter",
      "skip-l",
    ) +
    '<input type="range" id="scrub" dir="' +
    prefs.direction +
    '" min="1" max="' +
    c.pages.length +
    '" value="' +
    (i + 1) +
    '" aria-label="Page"><output>' +
    (i + 1) +
    " / " +
    c.pages.length +
    "</output>" +
    ib(
      rtl ? "prev-ch" : "next-ch",
      rtl ? "Previous chapter" : "Next chapter",
      "skip-r",
    ) +
    "</footer></main>"
  );
}
function setWipe(value) {
  const image = $(".overlay");
  if (image) image.parentElement.style.setProperty("--divider", value + "%");
  if (image) image.style.clipPath = "inset(0 " + (100 - value) + "% 0 0)";
}
// Keep at most the next screen in this signed-in tab. Reuse the image element
// on a turn so private pages never need a persistent browser cache.
const preparedPages = new Map();
function keepPrepared(sources) {
  for (const [src, img] of preparedPages) {
    if (!sources.includes(src)) {
      img.removeAttribute("src");
      preparedPages.delete(src);
    }
  }
}
function prepareNext(book, chapter, page, canvas) {
  if (canvas.classList.contains("continuous") || navigator.connection?.saveData) {
    keepPrepared([]);
    return;
  }
  const spread = prefs.layout === "spread" && view !== "compare";
  const step = spread ? 2 : 1;
  const index = chapter.pages.indexOf(page) + step;
  const nextChapter = index < chapter.pages.length ? chapter : book.chapters[book.chapters.indexOf(chapter) + 1];
  const start = nextChapter === chapter ? index : 0;
  const next = nextChapter?.pages.slice(start, start + step) || [];
  const sources = next.flatMap(p => view === "compare" && p.ready ? [best(p), url(p, "original")] :
    [view === "original" ? url(p, "original") : best(p)]);
  keepPrepared(sources);
  const visible = [...canvas.querySelectorAll(".page-image")];
  Promise.allSettled(visible.map(img => img.decode())).then(() => {
    if (!canvas.isConnected || canvas !== $(".canvas") || !config) return;
    for (const src of sources) {
      if (preparedPages.has(src)) continue;
      const img = new Image();
      img.fetchPriority = "low";
      img.onerror = () => { if (preparedPages.get(src) === img) preparedPages.delete(src); };
      preparedPages.set(src, img);
      img.src = src;
    }
  });
}
function mountReader(markup) {
  if (!preparedPages.size) {
    root.innerHTML = markup;
    return;
  }
  const template = document.createElement("template");
  template.innerHTML = markup;
  const replacements = [];
  for (const target of template.content.querySelectorAll(".page-image")) {
    const src = target.getAttribute("src");
    const img = preparedPages.get(src);
    if (!img) continue;
    preparedPages.delete(src);
    img.onerror = null;
    img.fetchPriority = "auto";
    for (const attr of target.attributes) {
      if (attr.name !== "src") img.setAttribute(attr.name, attr.value);
    }
    target.removeAttribute("src");
    replacements.push([target, img]);
  }
  root.replaceChildren(template.content);
  for (const [target, img] of replacements) target.replaceWith(img);
}
let lastRender = "",
  lastFocusPage = "";
function render() {
  if (!config) return;
  const { screen, book, chapter, page } = current(),
    reading = screen === "read" && page;
  const searchFocused = document.activeElement?.id === "search",
    caret = searchFocused ? $("#search").selectionStart : 0;
  const wipeValue = $("#wipe")?.value;
  const oldCanvas = $(".canvas"),
    oldScroll = oldCanvas?.scrollTop,
    same = lastRender === location.hash;
  const expanded = same && screen === "book" ? [...root.querySelectorAll("details[open] .pages")].map(el => el.dataset.chapter) : [];
  const hidden = same && $(".reader")?.classList.contains("clean");
  clearTimeout(hideTimer);
  if (reading) mountReader(reader(book, chapter, page));
  else {
    keepPrepared([]);
    root.innerHTML = screen === "settings"
      ? preferences()
      : screen === "book" && book
        ? detail(book)
        : library();
  }
  document.title =
    (reading
      ? book.title + " · " + chapter.title
      : screen === "settings"
        ? "Settings"
        : book?.title || "Library") + " · Tralens";
  lastRender = location.hash;
  for (const id of expanded) {
    const list = root.querySelector('.pages[data-chapter="' + id + '"]');
    if (list) list.parentElement.open = true;
  }
  if (searchFocused && $("#search")) {
    $("#search").focus();
    $("#search").setSelectionRange(caret, caret);
  }
  if (reading) {
    if (same && wipeValue && $("#wipe")) $("#wipe").value = wipeValue;
    setWipe($("#wipe")?.value || 50);
    if (hidden) $(".reader").classList.add("clean");
    else showChrome();
    const canvas = $(".canvas");
    prepareNext(book, chapter, page, canvas);
    const keepScroll = same && oldCanvas?.className === canvas.className;
    if (keepScroll) canvas.scrollTop = oldScroll;
    if (canvas.classList.contains("continuous")) {
      const target = canvas.querySelector('[data-page="' + page.id + '"]');
      if (!keepScroll) target?.scrollIntoView({ block: "start" });
      let timer;
      canvas.onscroll = () => {
        clearTimeout(timer);
        timer = setTimeout(() => {
          if (!canvas.isConnected) return;
          const middle =
            canvas.getBoundingClientRect().top + canvas.clientHeight * 0.45;
          const el = [...canvas.children].find(
            (x) => x.getBoundingClientRect().bottom > middle,
          );
          if (el && el.dataset.page !== current().page?.id) {
            const { book, chapter } = current();
            const next = chapter.pages.find((x) => x.id === el.dataset.page);
            history.replaceState(null, "", readURL(book, next));
            lastRender = location.hash;
            $("#scrub").value = chapter.pages.indexOf(next) + 1;
            $(".reader output").textContent =
              $("#scrub").value + " / " + chapter.pages.length;
            persistPosition(book, next);
            $("#page-status").innerHTML = pageBanner(next);
            updateEndLink(book, next);
          }
        }, 180);
      };
    }
    updateEndLink(book, page);
    persistPosition(book, page);
  } else lastFocusPage = "";
}
function persistPosition(book, page) {
  if (lastFocusPage === page.id) return;
  lastFocusPage = page.id;
  pendingPosition = pendingPosition
    .then(() => api("library/position", { book: book.id, page: page.id }))
    .catch((e) => toast(e.message));
}
function showChrome() {
  const r = $(".reader");
  if (!r) return;
  r.classList.remove("clean");
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => {
    if (!sheet.open && !r.querySelector(":focus-visible"))
      r.classList.add("clean");
  }, 4300);
}
function updateEndLink(book, page) {
  const forward = $('.edge[data-action="next"], .edge[data-action="chapters"]');
  const atEnd = book.chapters.at(-1).pages.at(-1).id === page.id ||
    (prefs.layout === "spread" && view !== "compare" && book.chapters.at(-1).pages.at(-2)?.id === page.id);
  forward.dataset.action = atEnd ? "chapters" : "next";
  forward.textContent = atEnd ? "Back to chapters" : "";
  if (atEnd) forward.setAttribute("aria-label", "Back to chapters");
  else {
    forward.setAttribute("aria-label", "Next page");
    forward.innerHTML = icon(prefs.direction === "rtl" ? "chev-l" : "chev-r");
  }
}
function goToPage(book, page) {
  const from = current();
  const previous = $(".canvas");
  history.replaceState(null, "", readURL(book, page));
  if (previous?.classList.contains("continuous") && current().chapter === from.chapter) {
    previous.querySelector('[data-page="' + page.id + '"]').scrollIntoView({block: "start"});
    lastRender = location.hash;
    const index = from.chapter.pages.indexOf(page) + 1;
    $("#scrub").value = index;
    $(".reader output").textContent = index + " / " + from.chapter.pages.length;
    $("#page-status").innerHTML = pageBanner(page);
    updateEndLink(book, page);
    persistPosition(book, page);
    return;
  }
  const top = previous?.scrollTop;
  render();
  const images = [...$(".canvas").querySelectorAll(".page-image")];
  if (previous && !previous.classList.contains("continuous") && !$(".continuous") &&
      images.some(img => !img.complete)) {
    previous.querySelectorAll("[id]").forEach(el => el.removeAttribute("id"));
    previous.inert = true;
    previous.setAttribute("aria-hidden", "true");
    previous.classList.add("previous-page");
    $(".reader").append(previous);
    previous.scrollTop = top;
    Promise.allSettled(images.map(img => img.decode())).then(() => previous.remove());
  }
}
function turn(delta, chapterOnly = false) {
  const { book, chapter, page } = current();
  if (!page) return;
  if (chapterOnly) {
    const c = book.chapters[book.chapters.indexOf(chapter) + delta];
    if (c) return goToPage(book, c.pages[0]);
    return;
  }
  const pages = pagesOf(book);
  let offset = delta;
  if (prefs.layout === "spread" && view !== "compare") {
    const index = chapter.pages.indexOf(page);
    if (delta > 0) {
      offset = Math.min(2, chapter.pages.length - index);
    } else if (index > 0) {
      offset = -Math.min(2, index);
    } else {
      const previous = book.chapters[book.chapters.indexOf(chapter) - 1];
      offset = previous?.pages.length % 2 === 0 ? -2 : -1;
    }
  }
  const n = pages[pages.indexOf(page) + offset];
  if (n) return goToPage(book, n);

}
function modal(title, html) {
  if (!sheet.open) focusReturn = document.activeElement;
  sheet.innerHTML =
    '<div class="dialog-in"><div class="line"><h2>' +
    esc(title) +
    "</h2>" +
    ib("close", "Close", "x") +
    "</div>" +
    html +
    "</div>";
  if (!sheet.open) sheet.showModal();
}
function closeSheet() {
  sheet.close();
}
sheet.addEventListener("close", () => {
  if (draft) {
    draft.files.forEach((f) => URL.revokeObjectURL(f.url));
    draft = null;
  }
  focusReturn?.focus();
  showChrome();
});
function addBook(id = "", files = []) {
  if (pendingImport) {
    $("#import-status").scrollIntoView();
    return toast("Finish or dismiss the current import first.");
  }
  draft = {
    book: id,
    title: "",
    chapter: id
      ? "Chapter " +
        (catalog.books.find((b) => b.id === id).chapters.length + 1)
      : "Chapter 1",
    files: [],
  };
  addFiles(files);
  drawAdd();
  if (!files.length && window.Capacitor?.isNativePlatform?.()) $("#files").click();
}
function addFiles(files) {
  draft.fileError = "";
  for (const file of [...files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { numeric: true }),
  )) {
    if (
      !/\.(png|jpe?g|webp|pdf|cbz|zip)$/i.test(file.name) ||
      (!/\.(pdf|cbz|zip)$/i.test(file.name) && file.size > config.max_upload_bytes)
    ) {
      draft.fileError = "Choose PNG, JPEG, WebP, PDF, CBZ or ZIP. Images can be up to 32 MiB each.";
      continue;
    }
    draft.files.push({ file, url: URL.createObjectURL(file) });
  }
  if (!draft.title && draft.files.length)
    draft.title = draft.files[0].file.name
      .replace(/\.[^.]+$/, "")
      .replace(/[_-]+/g, " ");
}
function drawAdd() {
  const selected = draft.files.length;
  const destination = catalog.books.find(b => b.id === draft.book)?.title || draft.title;
  const editing = sheet.open && $("#import .import-summary") && $("#import details")?.open;
  modal(
    draft.book ? "Add chapter" : "Add manga",
    '<form id="import" class="fields">' +
      (selected ? '<div class="import-summary"><b>' + esc(draft.files[0].file.name) + '</b>' +
        (selected > 1 ? ' + ' + (selected - 1) + ' more' : '') + '<p>' + esc(destination) +
        ' · ' + esc(draft.chapter) + '</p></div>' : '') +
      '<details' + (!selected || editing ? ' open' : '') + '><summary>' +
      (selected ? 'Edit details' : 'Choose files and destination') + '</summary><div class="fields"><div class="drop">' +
      icon("upload") +
      "<b>Drop manga pages here</b><small>PNG, JPEG, WebP, PDF, CBZ or ZIP · Up to 4 GiB per import</small>" +
      btn("choose", "Choose pages") +
      '<input id="files" type="file" accept="image/png,image/jpeg,image/webp,application/pdf,application/zip,application/x-zip-compressed,application/vnd.comicbook+zip,application/x-cbz,.cbz,.zip" multiple hidden></div>' +
      (draft.files.length
        ? "<div><small>" +
          draft.files.length +
          (draft.files.length === 1 ? " file" : " files") + ' · Reading order</small><div class="tray">' +
          draft.files
            .map(
              (f, i) =>
                '<div class="thumb" draggable="true" data-index="' +
                i +
                '">' +
                (/\.(pdf|cbz|zip)$/i.test(f.file.name) ? '<span class="pdf-thumb">' + esc(f.file.name.split(".").pop().toUpperCase()) + '</span>' : '<img src="' + f.url + '" alt="">') +
                '<small>' +
                esc(f.file.name) +
                "</small><div>" +
                ["←", "×", "→"]
                  .map(
                    (l, x) =>
                      '<button type="button" aria-label="' +
                      ["Move earlier", "Remove page", "Move later"][x] +
                      '" data-item="' +
                      i +
                      '" data-move="' +
                      (x - 1) +
                      '">' +
                      l +
                      "</button>",
                  )
                  .join("") +
                "</div></div>",
            )
            .join("") +
          "</div></div>"
        : "") +
      '<label class="field">Add to<select id="target"><option value="">New manga</option>' +
      catalog.books
        .map(
          (b) =>
            '<option value="' +
            b.id +
            '" ' +
            (draft.book === b.id ? "selected" : "") +
            ">" +
            esc(b.title) +
            "</option>",
        )
        .join("") +
      "</select></label>" +
      (!draft.book
        ? '<label class="field">Title<input id="title" maxlength="200" required value="' +
          esc(draft.title) +
          '"></label>'
        : "") +
      '<label class="field">Chapter<input id="chapter" maxlength="200" required value="' +
      esc(draft.chapter) +
      '"></label>' +
      '</div></details><p class="hint">' +
      esc(privacy()) +
      '</p><div class="error" role="alert" id="import-error">' + esc(draft.fileError) + '</div><button class="primary" ' +
      (!draft.files.length || !config.render_enabled ? "disabled" : "") +
      (prefs.translator === "local" ? ">Add &amp; translate</button>" : ">Add &amp; translate with ChatGPT</button>") +
      (!config.render_enabled
        ? '<p class="error">Rendering is unavailable. Start the backend with a font manifest.</p>'
        : "") +
      "</form>",
  );
}
function reorder(i, j) {
  if (j < 0 || j >= draft.files.length) return;
  draft.files.splice(j, 0, draft.files.splice(i, 1)[0]);
  drawAdd();
}
async function retry(book, page) {
  if (prefs.translator !== "local" &&
      !window.confirm("Translate again with ChatGPT? Tralens will send the selected pages and detected text to OpenAI and process the queue in the background.")) return;
  await api("library/retry", {
    book: book.id,
    page: page?.id,
    options: settings(),
  });
  closeSheet();
  dismissed = "";
  await refresh(true);
  toast("Queued for translation.");
}
function menu(kind) {
  const { book, page } = current();
  if (kind === "reader-prefs")
    return modal(
      "Reader settings",
      '<div class="fields">' +
        readingFields.map(field).join("") +
        field([
          "comparison",
          "Compare pages",
          [
            ["swipe", "Swipe divider"],
            ["side", "Side by side"],
          ],
        ]) +
        "</div>",
    );
  if (kind === "book-menu")
    return modal(
      book.title,
      '<div class="menu-list">' +
        btn("rename", "Rename manga") +
        btn("add-chapter", "Add chapter") +
        btn("retry-book", "Translate again") +
        btn("delete", "Delete from library", "danger") +
        "</div>",
    );
  if (kind === "download-book") {
    const pages = pagesOf(book);
    const ready = pages.filter(p => p.ready).length;
    return modal(
      "Download manga",
      '<div class="menu-list">' +
        ["translated", "original", "both"].filter(v => v !== "translated" || ready)
          .map(
            (v) =>
              '<a class="btn" href="/api/library/books/' +
              book.id +
              "/" +
              v +
              '.zip" download="' + esc(book.title + "-" + v + ".zip") + '">' +
              esc(
                {
                  translated: "Translated pages (" + ready + " of " + pages.length + ")",
                  original: "Original pages",
                  both: "All originals + " + ready + " translated",
                }[v],
              ) +
              " · ZIP</a>",
          )
          .join("") +
        "</div>",
    );
  }
  if (kind === "rename")
    return modal(
      "Rename manga",
      '<form id="rename" class="fields"><label class="field">Title<input id="new-title" required maxlength="200" value="' +
        esc(book.title) +
        '"></label><button class="primary">Save title</button></form>',
    );
  if (kind === "delete")
    return modal(
      "Delete " + book.title + "?",
      '<p>This removes its chapters, page images and reading position from your library.</p><div class="actions">' +
        btn("close", "Cancel") +
        btn("delete-confirm", "Delete manga", "danger") +
        "</div>",
    );
  if (kind === "page-menu") {
    const {chapter} = current();
    const index = chapter.pages.indexOf(page);
    const shown = prefs.layout === "spread" && view !== "compare" ? chapter.pages.slice(index, index + 2) : [page];
    const transcripts = shown.filter(p => p.has_text).map(p =>
      '<button type="button" data-action="transcript" data-id="' + esc(p.id) + '">Read English' +
      (shown.length > 1 ? ' · Page ' + (chapter.pages.indexOf(p) + 1) : '') + '</button>').join("");
    return modal(
      "Page actions",
      '<div class="menu-list"><a class="btn" href="' +
        url(page, "original") +
        '" download="' +
        esc(page.name) +
        '">Download original</a>' +
        (page.ready
          ? '<a class="btn" href="' +
            best(page) +
            '" download="translated.png">Download translation</a>'
          : "") +
        transcripts +
        (["queued", "running", "waiting"].includes(page.status) ? btn("skip-page", "Skip translation") : btn("retry-page", "Translate again")) +
        btn("fullscreen", "Toggle full screen") +
        "</div>",
    );
  }
}
async function openReading(pageId, point) {
  const page = pagesOf(current().book).find(p => p.id === pageId);
  if (!page) return;
  const locationAtTap = location.hash;
  const lines = await api("library/pages/" + page.id + "/transcript");
  if (location.hash !== locationAtTap) return;
  const index = point ? lines.findIndex(({box: [left, top, right, bottom]}) =>
    point.x >= left && point.x <= right && point.y >= top && point.y <= bottom) : -1;
  if (point && index < 0) {
    if ($(".reader").classList.contains("clean")) showChrome();
    else $(".reader").classList.add("clean");
    return;
  }
  modal("Page text", '<ol class="reading-text">' + lines.map(({box, text}) => {
    const [x, y, right, bottom] = box;
    const width = right - x + 8, height = bottom - y + 8;
    const region = [x - 4, y - 4, width, height].join(" ");
    return '<li><svg aria-hidden="true" viewBox="' + esc(region) +
      '" style="aspect-ratio:' + width + '/' + height + '"><image href="' + url(page, "original") +
      '" width="' + page.width + '" height="' + page.height + '"/></svg><p>' + esc(text) + '</p></li>';
  }).join("") + '</ol>');
  if (index >= 0) sheet.querySelectorAll(".reading-text li")[index].scrollIntoView({block:"center"});
}
async function act(action, pageId) {
  if (action === "transcript") return openReading(pageId);
  if (action === "skip-page") {
    const {book, page} = current();
    await api("library/skip", {book: book.id, page: page.id});
    closeSheet();
    return refresh(true);
  }
  if (action === "owner-logout") {
    const reply = await fetch("/api/auth/logout", {method: "POST"});
    if (reply.ok || reply.status === 401) accountChanged();
    else toast("Could not sign out.");
    return;
  }
  if (action === "configure-server") return showServerForm();
  const { book, page } = current();
  if (
    [
      "book-menu",
      "reader-prefs",
      "page-menu",
      "delete",
      "rename",
      "download-book",
    ].includes(action)
  )
    return menu(action);
  if (["translated", "original", "compare"].includes(action)) {
    view = action;
    return render();
  }
  if (action === "toggle-comparison") {
    prefs.comparison = prefs.comparison === "side" ? "swipe" : "side";
    save();
    render();
    return;
  }
  if (action === "dismiss") {
    dismissed = page.id + ":" + page.status;
    return render();
  }
  if (action === "close") return closeSheet();
  if (action === "theme") {
    prefs.theme =
      document.documentElement.dataset.theme === "dark" ||
      (prefs.theme === "system" &&
        matchMedia("(prefers-color-scheme: dark)").matches)
        ? "light"
        : "dark";
    save();
    return;
  }
  if (action === "add" || action === "add-chapter")
    return addBook(action === "add-chapter" ? book.id : "");
  if (action === "chapters") return location.replace("#book/" + book.id);
  if (action === "choose") return $("#files").click();
  if (action === "next" || action === "prev")
    return turn(action === "next" ? 1 : -1);
  if (action === "next-ch" || action === "prev-ch")
    return turn(action === "next-ch" ? 1 : -1, true);
  if (action === "disconnect-chatgpt") {
    if (!window.confirm("Disconnect the ChatGPT account from this server? Queued ChatGPT translations may fail and can be retried with Local.")) return;
    await api("chatgpt/disconnect", {});
    config.chatgpt_connected = false;
    config.chatgpt_models = null;
    prefs.translator = "local";
    save();
    render();
    toast("ChatGPT disconnected.");
    return;
  }
  if (action === "retry-page" || action === "retry-book")
    return retry(book, action === "retry-page" ? page : null);
  if (action === "delete-confirm") {
    await api("library/delete", { book: book.id });
    closeSheet();
    location.hash = "library";
    await refresh(true);
  }
  if (action === "fullscreen") {
    closeSheet();
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  }
  if (action === "reconnect") return init();
}
document.addEventListener("click", (e) => {
  const download = e.target.closest("a[download]");
  if (download && window.Capacitor?.isPluginAvailable?.("ReaderConnection")) {
    e.preventDefault();
    const path = new URL(download.href).pathname;
    api("library/download-ticket", {path})
      .then(({ticket}) => window.Capacitor.Plugins.ReaderConnection.download({
        path, ticket, filename: download.download
      }))
      .then(() => toast("Download started."))
      .catch(error => toast(error.message));
    return;
  }
  const a = e.target.closest("[data-action]");
  if (a) {
    act(a.dataset.action, a.dataset.id).catch((err) => toast(err.message));
    return;
  }
  const m = e.target.closest("[data-item]");
  if (m) {
    const i = Number(m.dataset.item),
      delta = Number(m.dataset.move);
    if (delta) reorder(i, i + delta);
    else {
      URL.revokeObjectURL(draft.files.splice(i, 1)[0].url);
      drawAdd();
    }
    return;
  }
  const original = e.target.closest("[data-reading-fallback] .page-image");
  if (original) {
    const bounds = original.getBoundingClientRect();
    openReading(original.parentElement.dataset.page, {
      x: (e.clientX - bounds.left) * original.naturalWidth / bounds.width,
      y: (e.clientY - bounds.top) * original.naturalHeight / bounds.height,
    }).catch(err => toast(err.message));
    return;
  }
  if (e.target.closest(".canvas") && !e.target.closest("input,.wipe")) {
    const r = $(".reader");
    if (r.classList.contains("clean")) showChrome();
    else r.classList.add("clean");
  }
});
document.addEventListener("input", (e) => {
  if (e.target.id === "search") {
    query = e.target.value;
    $("#shelf").innerHTML = shelf();
  }
  if (e.target.id === "wipe") setWipe(e.target.value);
  if (draft && ["title", "chapter"].includes(e.target.id)) {
    draft[e.target.id] = e.target.value;
    const summary = $(".import-summary p");
    if (summary) {
      const book = catalog.books.find(b => b.id === draft.book);
      summary.textContent = (book?.title || draft.title) + " · " + draft.chapter;
    }
  }
});
function showChatGPTConsent(model) {
  const phonePlugin = window.Capacitor?.isPluginAvailable?.("PhoneSignIn");
  const androidBrowser = (/Android/i.test(navigator.userAgent) || navigator.userAgentData?.platform === "Android") && !phonePlugin;
  if (androidBrowser) {
    modal("Connect on your phone",
      '<p>Open Tralens and choose a GPT model in Settings → Translation to sign in.</p>' +
      '<a id="open-tralens" class="btn primary" href="intent://signin/return#Intent;scheme=tralens-reader;package=app.tralens.reader;end">Open Tralens</a>' +
      '<div role="status" id="chatgpt-connect"></div>');
    const panel = $("#chatgpt-connect");
    const onReturn = async () => {
      if (document.hidden || !sheet.open || !panel.isConnected) return;
      try {
        const result = await api("chatgpt/models");
        if (sheet.open && panel.isConnected && result.state === "connected") useChatGPTModel(model, result);
      } catch (err) {
        if (panel.isConnected) panel.textContent = err.message;
      }
    };
    $("#open-tralens").onclick = () => document.addEventListener("visibilitychange", onReturn);
    sheet.addEventListener("close", () => document.removeEventListener("visibilitychange", onReturn), {once: true});
    return;
  }
  modal("Connect to ChatGPT",
    '<div id="chatgpt-consent"><p>GPT translations send selected pages and text to OpenAI using your ChatGPT plan.</p>' +
    '<details><summary>Privacy details</summary><p>Your server stores your account ID, email and sign-in tokens. Translations can finish in the background. Disconnect anytime in Settings.</p>' +
    '<a href="/privacy" target="_blank" rel="noopener">Tralens privacy notice</a> · <a href="https://openai.com/policies/privacy-policy/" target="_blank" rel="noopener">OpenAI Privacy Policy</a></details>' +
    '<button id="consent-chatgpt" class="primary" type="button">Continue with ChatGPT</button></div>' +
    '<div id="chatgpt-connect" role="status"></div>');
  $("#consent-chatgpt").onclick = () => {
    $("#chatgpt-consent").remove();
    selectChatGPT(model, $("#chatgpt-connect"));
  };
}

function useChatGPTModel(model, result) {
  config.chatgpt_connected = true;
  config.chatgpt_models = result.models;
  if (!result.models.includes(model)) throw Error("Your ChatGPT account does not offer this model.");
  prefs.translator = model;
  save();
  closeSheet();
  render();
}

async function selectChatGPT(model, panel) {
  panel.textContent = "Checking your connection…";
  const active = () => sheet.open && panel.isConnected && config;
  try {
    let result;
    if (window.Capacitor?.isPluginAvailable?.("PhoneSignIn")) {
      const phone = window.Capacitor.Plugins.PhoneSignIn;
      const {port} = await phone.prepare();
      const cancel = () => phone.cancel({port}).catch(() => {});
      sheet.addEventListener("close", cancel, {once: true});
      try {
        if (!active()) return;
        result = await api("chatgpt/phone/start", {port});
        if (!active()) return;
        if (result.state === "authorize") {
          panel.textContent = "Finish signing in, then tap Open Tralens.";
          const fields = await phone.authorize({url: result.url});
          if (!active()) return;
          result = await api("chatgpt/phone/finish", fields);
        }
      } finally {
        sheet.removeEventListener("close", cancel);
        cancel();
      }
    } else {
      result = await api("chatgpt/connect", {});
    }
    let displayedURL;
    while (active() && result.state === "pending") {
      if (result.url && displayedURL !== result.url) {
        const url = new URL(result.url);
        if (url.protocol !== "http:" || url.hostname !== "127.0.0.1") throw Error("Invalid sign-in address.");
        displayedURL = result.url;
        const remote = config.owner_mode && !["localhost", "127.0.0.1"].includes(location.hostname);
        panel.innerHTML = remote
          ? '<p>Finish signing in on your server.</p>' +
            '<details><summary>Use an SSH tunnel</summary><p>Forward port ' + esc(url.port) +
            ' from your computer to the same loopback port on the server over SSH, then open this link on your computer.</p>' +
            '<a href="' + esc(url.href) + '" target="_blank" rel="noopener noreferrer">Continue with ChatGPT</a></details>'
          : '<a class="btn primary" href="' + esc(url.href) + '" target="_blank" rel="noopener noreferrer">Continue with ChatGPT</a>';
      }
      await new Promise(resolve => setTimeout(resolve, 800));
      if (!active()) return;
      result = await api("chatgpt/status");
      if (result.state === "connected") result = await api("chatgpt/connect", {});
    }
    if (!active()) return;
    if (result.state !== "connected") throw Error(result.error || "Sign-in did not finish.");
    useChatGPTModel(model, result);
  } catch (err) {
    if (active()) panel.textContent = err.message;
  }
}
document.addEventListener("change", async (e) => {
  const el = e.target;
  if (el.dataset.pref) {
    if (el.dataset.pref === "translator" && el.value !== "local" && config.frontier_provider === "chatgpt" &&
        !(config.chatgpt_connected && config.chatgpt_models?.includes(el.value))) {
      const requested = el.value;
      el.value = prefs.translator;
      if (config.chatgpt_connected) {
        let result;
        try { result = await api("chatgpt/models"); } catch (err) { toast(err.message); return; }
        if (result.state === "connected") {
          config.chatgpt_models = result.models;
          if (result.models.includes(requested)) prefs.translator = requested;
          else toast("This model is unavailable for your account.");
          save();
          render();
        } else showChatGPTConsent(requested);
      } else showChatGPTConsent(requested);
      return;
    }
    prefs[el.dataset.pref] = el.value;
    save();
    render();
  }
  if (el.id === "scrub") {
    const { book, chapter } = current();
    await goToPage(book, chapter.pages[Number(el.value) - 1]);
  }
  if (el.id === "files") {
    addFiles(el.files);
    drawAdd();
  }
  if (el.id === "target") {
    draft.book = el.value;
    draft.chapter =
      "Chapter " +
      ((catalog.books.find((b) => b.id === el.value)?.chapters.length || 0) +
        1);
    drawAdd();
  }
});
function importStatus(message, percent = null, failed = false) {
  $("#import-status").hidden = false;
  $("#import-name").textContent = pendingImport.name;
  $("#import-message").textContent = message;
  const progress = $("#import-progress");
  progress.hidden = failed;
  if (percent === null) progress.removeAttribute("value");
  else progress.value = percent;
  $("#import-stop").hidden = failed || !pendingImport.request;
  $("#import-retry").hidden = $("#import-dismiss").hidden = !failed;
}

function clearImport() {
  pendingImport?.request?.abort();
  pendingImport = null;
  $("#import-status").hidden = true;
}

function sendImport(url, opts) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const item = pendingImport;
    item.request = xhr;
    $("#import-stop").hidden = false;
    $("#import-stop").textContent = "Stop upload";
    xhr.open("POST", url);
    xhr.setRequestHeader("X-Tralens-Token", opts.headers["X-Tralens-Token"]);
    let timer;
    function watch(ms) {
      clearTimeout(timer);
      timer = setTimeout(() => xhr.abort(), ms);
    }
    xhr.upload.onprogress = event => {
      watch(90000);
      if (event.lengthComputable) {
        const percent = Math.round(100 * event.loaded / event.total);
        importStatus("Uploading · " + percent + "%", percent);
      }
    };
    xhr.upload.onload = () => {
      $("#import-stop").textContent = "Stop waiting";
      watch(6 * 60000);
      importStatus("Upload sent. Preparing pages on your server…");
    };
    xhr.onload = () => resolve(new Response(xhr.responseText || null, {status: xhr.status}));
    xhr.onerror = () => reject(Error("Connection lost. Reconnect and retry."));
    xhr.onabort = () => reject(Error(item.stopped ? "Stopped waiting. Retry checks whether your chapter was saved." :
      "Your server stopped responding. Check your connection and retry."));
    xhr.onloadend = () => clearTimeout(timer);
    watch(90000);
    xhr.send(opts.body);
  });
}

async function runImport(item, retry = false) {
  item.request = null;
  item.stopped = false;
  importStatus(retry ? "Checking your library…" : "Uploading…");
  const receipt = () => api("library/imports/" + item.id);
  let added;
  try {
    if (retry) added = await receipt();
    if (pendingImport !== item) return;
    if (!added) {
      importStatus("Uploading…");
      added = await api("library/add", item.form, true, sendImport);
    }
  } catch (err) {
    if (pendingImport !== item) return;
    // A disconnect after the upload can hide a successful commit.
    item.request = null;
    importStatus("Checking whether your chapter was saved…");
    try { added = await receipt(); } catch { /* Keep the files available for retry. */ }
    if (pendingImport !== item) return;
    if (!added) {
      const network = err instanceof TypeError || err.name === "TimeoutError";
      importStatus(network ? "Couldn’t reach your server. Reconnect and retry." : err.message, null, true);
      return;
    }
  }
  if (pendingImport !== item) return;
  clearImport();
  toast("Added to your library.");
  try {
    await refresh(true);
  } catch {
    if (config) toast("Added to your library. Reconnect to see your pages.");
    schedulePoll();
    return;
  }
  const book = catalog.books.find(b => b.id === added.book);
  const chapter = book?.chapters.find(c => c.id === added.chapter);
  if (chapter?.pages.length && location.hash === item.route && !sheet.open) {
    history.replaceState(null, "", "#book/" + book.id);
    location.hash = readURL(book, chapter.pages[0]);
  }
}
$("#import-stop").onclick = () => {
  pendingImport.stopped = true;
  pendingImport.request?.abort();
};
$("#import-retry").onclick = () => runImport(pendingImport, true);
$("#import-dismiss").onclick = clearImport;

document.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (e.target.id === "account-form") return;
  if (e.target.id === "import") {
    if (pendingImport) return;
    const id = Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, "0")).join("");
    const form = new FormData();
    form.append("metadata", JSON.stringify({import_id: id, book: draft.book,
      title: draft.title, chapter: draft.chapter, options: settings()}));
    draft.files.forEach(f => form.append("page", f.file));
    pendingImport = {id, form, name: draft.title || draft.chapter, route: location.hash};
    closeSheet();
    return runImport(pendingImport);
  }
  const button = e.target.querySelector('button:not([type="button"])');
  if (button) button.disabled = true;
  try {
    if (e.target.id === "rename") {
      await api("library/rename", {
        book: current().book.id,
        title: $("#new-title").value,
      });
      closeSheet();
      await refresh(true);
    }
  } catch (err) {
    toast(err.message);
  } finally {
    if (button) button.disabled = false;
  }
});
let dragged = -1;
document.addEventListener("dragstart", (e) => {
  dragged = Number(e.target.closest("[data-index]")?.dataset.index ?? -1);
});
document.addEventListener("dragover", (e) => {
  if (draft || e.dataTransfer.types.includes("Files")) e.preventDefault();
});
document.addEventListener("drop", (e) => {
  e.preventDefault();
  if (!config) return;
  if (e.dataTransfer.files.length) {
    if (!draft) addBook("", e.dataTransfer.files);
    else {
      addFiles(e.dataTransfer.files);
      drawAdd();
    }
  } else if (draft && dragged >= 0) {
    const slot = e.target.closest("[data-index]");
    if (slot) reorder(dragged, Number(slot.dataset.index));
  }
  dragged = -1;
});
document.addEventListener("keydown", (e) => {
  if (
    sheet.open ||
    !current().page ||
    e.target.matches("input,select,textarea")
  )
    return;
  if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
    e.preventDefault();
    turn((e.key === "ArrowLeft") === (prefs.direction === "rtl") ? 1 : -1);
  }
  if (e.key === "Escape") location.hash = "book/" + current().book.id;
});
let touch;
document.addEventListener(
  "touchstart",
  (e) => {
    touch =
      e.target.closest(".canvas") && !e.target.closest("input,.wipe")
        ? e.touches[0]
        : null;
  },
  { passive: true },
);
document.addEventListener(
  "touchend",
  (e) => {
    if (!touch) return;
    const dx = e.changedTouches[0].clientX - touch.clientX,
      dy = e.changedTouches[0].clientY - touch.clientY;
    touch = null;
    if (Math.abs(dx) > 65 && Math.abs(dx) > Math.abs(dy) * 1.6)
      turn(dx > 0 === (prefs.direction === "rtl") ? 1 : -1);
  },
  { passive: true },
);
addEventListener("resize", () => { if (config && route()[0] === "read" && !sheet.open) render(); });
addEventListener("tralens-back", event => {
  if (sheet.open) closeSheet();
  else if ($("#keep-server")) $("#keep-server").click();
  else if ($('#account-form input[autocomplete="new-password"]') && $(".account-switch:not([hidden])")) $(".account-switch").click();
  else if (current().page) location.replace("#book/" + current().book.id);
  else if (config && ["book", "settings"].includes(route()[0])) location.replace("#library");
  else return;
  event.preventDefault();
});
addEventListener("hashchange", () => {
  if (!config) return;
  if (sheet.open) closeSheet();
  render();
  window.scrollTo(0, 0);
});
function accountChanged() {
  try { localStorage.setItem("tralens-account-change", crypto.randomUUID()); } catch {}
  location.reload();
}
addEventListener("storage", event => {
  if (event.key === "tralens-account-change") location.reload();
});
addEventListener("pageshow", event => {
  if (event.persisted) location.reload();
});
function serverOrigin(value) {
  const raw = value.trim();
  const scheme = /^(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/i.test(raw) ? "http://" : "https://";
  const url = new URL(raw.includes("://") ? raw : scheme + raw);
  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if ((url.protocol !== "https:" && !(loopback && url.protocol === "http:")) ||
      url.username || url.password || url.pathname !== "/" || url.search || url.hash)
    throw Error("Enter an HTTPS server address, or use HTTP on localhost.");
  return url.origin;
}
const nativeConnection = window.Capacitor?.isPluginAvailable?.("ReaderConnection")
  ? window.Capacitor.Plugins.ReaderConnection : null;
const serverKey = "tralens-verified-server";
let selectedServer = "";
try {
  const stored = localStorage.getItem(serverKey);
  if (stored) selectedServer = serverOrigin(stored);
  else if (localStorage.getItem("tralens-account-change")) selectedServer = location.origin;
} catch {}
if (nativeConnection) {
  try { if ((await nativeConnection.selected()).selected) selectedServer = location.origin; } catch {}
}
function rememberServer(origin) {
  selectedServer = origin;
  try { localStorage.setItem(serverKey, origin); } catch {}
}
function verifyServer(origin) {
  return new Promise((resolve, reject) => {
    const worker = new Worker("/verify-server.js");
    const finish = ok => {
      clearTimeout(timer);
      worker.terminate();
      if (ok) resolve();
      else reject(Error("Couldn’t verify this Tralens server. Check the address and connection, then try again."));
    };
    const timer = setTimeout(() => finish(false), 8000);
    worker.onmessage = event => finish(event.data === true);
    worker.onerror = () => finish(false);
    worker.postMessage(origin);
  });
}
function accountCard(title, intro, content) {
  keepPrepared([]);
  if (pendingImport) clearImport();
  if (sheet.open) sheet.close();
  clearTimeout(pollTimer);
  catalog = {books: [], bytes: 0};
  document.title = title + " · Tralens";
  root.innerHTML = `<main class="account-screen"><section class="account-card">
    <div class="account-brand"><span class="mark" aria-hidden="true">文<b>A</b></span>Tralens</div>
    <h1 tabindex="-1">${esc(title)}</h1><p class="account-intro">${esc(intro)}</p>${content}</section></main>`;
}
function showServerForm(message = "", initial = selectedServer || (nativeConnection ? "" : location.origin)) {
  config = null;
  accountCard("Connect to your server", "Verify the address of your Tralens server to continue.", `
    <form id="account-form" method="post">
      <label class="field">Server address<input name="server" required inputmode="url" autocomplete="url" autocapitalize="none" spellcheck="false" value="${esc(initial)}" placeholder="manga.example.com"></label>
      <p class="error" role="alert">${esc(message)}</p><button class="primary">Verify server</button>
    </form>${selectedServer === location.origin ? '<button type="button" class="account-switch" id="keep-server">Back to current server</button>' : ""}`);
  const form = $("#account-form"), input = form.elements.server, button = form.querySelector("button");
  const back = $("#keep-server");
  if (back) back.onclick = () => location.reload();
  form.onsubmit = async event => {
    event.preventDefault();
    if (button.disabled) return;
    const error = form.querySelector(".error");
    error.textContent = "";
    button.disabled = input.readOnly = true;
    button.textContent = "Verifying…";
    try {
      const origin = serverOrigin(input.value);
      await verifyServer(origin);
      if (!form.isConnected) return;
      if (nativeConnection && origin !== location.origin) {
        await nativeConnection.connect({url: origin});
      } else if (origin !== location.origin) {
        rememberServer(origin);
        location.assign(origin + "/#verify-server");
      } else {
        rememberServer(origin);
        await init();
        $(".account-card h1")?.focus();
      }
    } catch (err) {
      error.textContent = err.message;
      input.focus();
    } finally {
      button.disabled = input.readOnly = false;
      button.textContent = "Verify server";
    }
  };
}
function showAccountForm(message = "") {
  if ($("#account-form")) return;
  if (selectedServer === location.origin) showCredentials(message);
  else showServerForm(message);
}
function requestSetupLink(form, message) {
  $(".account-card h1").textContent = "Use your setup link";
  document.title = "Owner setup · Tralens";
  $(".account-intro").textContent = "Open the private link from your server, or paste it here to continue in the app.";
  form.innerHTML = `
    <label class="field">Private setup link<input name="link" type="url" required autocomplete="off" autocapitalize="none" spellcheck="false"></label>
    <p class="account-guidance">On the server, find the link in <code>/library/owner-setup.url</code> inside the reader container.</p>
    <p class="error" role="alert"></p><button class="primary">Continue setup</button>`;
  const alert = form.querySelector('[role="alert"]');
  alert.textContent = message;
  form.onsubmit = event => {
    event.preventDefault();
    const prefix = location.origin + "/#setup=";
    const link = form.elements.link.value.trim();
    const token = link.slice(prefix.length);
    if (!link.startsWith(prefix) || !/^[A-Za-z0-9_-]{43}$/.test(token)) {
      alert.textContent = "Paste the full setup link for this server, including #setup=.";
      return;
    }
    form.reset();
    showCredentials("", token);
    $('input[name="username"]')?.focus();
  };
}
function showCredentials(message = "", setupToken = "") {
  accountCard("Sign in", "Open your manga library.", `
    <div class="account-server"><span>${esc(location.origin)}</span><button type="button" id="change-server" aria-label="Change server">Change</button></div>
    <form id="account-form" method="post">
      <label class="field">Username<input name="username" required maxlength="32" autocomplete="username" autocapitalize="none" spellcheck="false"></label>
      <label class="field">Password<input name="password" type="password" required autocomplete="current-password"></label>
      <p class="account-guidance" hidden>Use a 3–32 character username and a password of at least 15 characters.</p>
      <p role="alert" class="error">${esc(message)}</p><button class="primary">Sign in</button>
    </form><button type="button" class="account-switch" hidden>Create an account</button>`);
  const form = $("#account-form"), password = form.elements.password;
  const submit = form.querySelector("button"), toggle = $(".account-switch"), alert = form.querySelector('[role="alert"]');
  const change = $("#change-server");
  change.onclick = () => { showServerForm(); $(".account-card h1").focus(); };
  let action = "login";
  function setMode(next) {
    action = next;
    const creating = next !== "login", owner = next === "setup";
    password.value = "";
    password.autocomplete = creating ? "new-password" : "current-password";
    password.minLength = creating ? 15 : 1;
    form.elements.username.pattern = creating ? "[a-zA-Z0-9][a-zA-Z0-9._\\-]{2,31}" : ".*";
    $(".account-guidance").hidden = !creating;
    submit.textContent = owner ? "Create owner account" : creating ? "Create account" : "Sign in";
    $(".account-card h1").textContent = owner ? "Create your owner account" : creating ? "Create an account" : "Sign in";
    document.title = $(".account-card h1").textContent + " · Tralens";
    $(".account-intro").textContent = owner ? "As owner, you can connect ChatGPT. Each reader gets a separate library." :
      creating ? "Start a private library on this server." : "Open your manga library.";
    toggle.textContent = creating ? "Back to sign in" : "Create an account";
  }
  fetch("/api/auth/settings", {redirect:"error"}).then(r => r.ok ? r.json() : null).then(data => {
    if (!form.isConnected || !data) return;
    if (data.setup_required) {
      if (data.setup_link_required && !setupToken) requestSetupLink(form, message);
      else setMode("setup");
    } else {
      toggle.hidden = !data.registration;
      if (setupToken) alert.textContent = "This server already has an owner. Sign in to continue.";
    }
  }).catch(() => {});
  toggle.onclick = () => {
    setMode(action === "register" ? "login" : "register");
    alert.textContent = "";
    form.elements.username.focus();
  };
  form.onsubmit = async event => {
    event.preventDefault();
    if (submit.disabled) return;
    submit.disabled = toggle.disabled = change.disabled = true;
    alert.textContent = "";
    try {
      const response = await fetch("/api/auth/" + action, {
        method: "POST", redirect: "error", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({username: form.elements.username.value, password: password.value,
          ...(action === "setup" && setupToken ? {setup_token: setupToken} : {})}),
      });
      if (!response.ok) {
        const reason = (await response.json()).error || "Sign-in failed.";
        if (action === "setup" && (response.status === 403 || response.status === 409)) {
          showCredentials(reason);
          return;
        }
        throw Error(reason);
      }
      password.value = "";
      rememberServer(location.origin);
      accountChanged();
    } catch (error) {
      alert.textContent = error.message;
    } finally {
      submit.disabled = toggle.disabled = change.disabled = false;
    }
  };
}
async function init() {
  root.innerHTML = '<main class="empty"><h1>Connecting to your library…</h1><p>' + esc(location.origin) + '</p></main>';
  try {
    config = await api("config");
    rememberServer(location.origin);
    try {
      const saved = localStorage.getItem("tralens-reader:" + config.username) ||
        (config.username === "owner" ? localStorage.getItem("tralens-reader") : null);
      prefs = {...defaults, ...JSON.parse(saved || "{}")};
      document.documentElement.dataset.theme = prefs.theme;
    } catch { prefs = {...defaults}; }
    if (prefs.translator !== "local" && (config.chatgpt_connected === false ||
        (config.chatgpt_models && !config.chatgpt_models.includes(prefs.translator)))) {
      prefs.translator = "local";
      save();
      toast("Local translation is selected. Choose a GPT model in Settings to connect.");
    }
    await refresh(true);
  } catch (err) {
    if ($("#account-form")) return;
    config = null;
    accountCard("Server unavailable", "Check your connection to " + location.host + " and try again.",
      '<button type="button" class="primary" id="retry-server">Retry connection</button>' +
      '<button type="button" class="account-switch" id="choose-server">Change server</button>');
    $("#retry-server").onclick = init;
    $("#choose-server").onclick = () => showServerForm();
  }
}
if (location.hash.startsWith("#setup=")) {
  const token = location.hash.slice(7);
  history.replaceState(null, "", location.pathname);
  rememberServer(location.origin);
  showCredentials("", token);
} else if (location.hash === "#verify-server") {
  history.replaceState(null, "", location.pathname);
  try {
    await verifyServer(location.origin);
    rememberServer(location.origin);
    await init();
  } catch (error) { showServerForm(error.message, location.origin); }
} else if (selectedServer && selectedServer !== location.origin && !nativeConnection) {
  accountCard("Connecting to your server", selectedServer,
    '<button type="button" class="account-switch" id="choose-server">Change server</button>');
  const change = $("#choose-server");
  change.onclick = () => showServerForm();
  try {
    await verifyServer(selectedServer);
    if (change.isConnected) location.replace(selectedServer + "/#verify-server");
  } catch (error) {
    if (change.isConnected) showServerForm(error.message, selectedServer);
  }
} else if (nativeConnection && !selectedServer) showServerForm();
else await init();
document.addEventListener("visibilitychange", () => {
  clearTimeout(pollTimer);
  if (!document.hidden) poll();
});

function dragDivider(e) {
  const area = e.target.closest(".wipe");
  if (!area || e.target.matches("input")) return;
  const box = area.getBoundingClientRect();
  const value = Math.max(
    0,
    Math.min(100, (100 * (e.clientX - box.left)) / box.width),
  );
  $("#wipe").value = value;
  setWipe(value);
}
document.addEventListener("pointerdown", (e) => {
  const area = e.target.closest(".wipe");
  if (area && $(".reader").classList.contains("clean")) showChrome();
  if (area && !e.target.matches("input")) {
    area.setPointerCapture(e.pointerId);
    dragDivider(e);
  }
});
document.addEventListener("pointermove", (e) => {
  if (e.target.closest(".wipe")?.hasPointerCapture(e.pointerId)) dragDivider(e);
});
