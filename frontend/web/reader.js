const $ = (selector) => document.querySelector(selector);
const root = $("#root"),
  sheet = $("#sheet");
const defaults = {
  translator: "local",
  paint: "classical",
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
  loading = false,
  focusReturn,
  dismissed = "",
  uploadBusy = false;
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
const url = (p, which) =>
  "/api/library/pages/" +
  p.id +
  "/" +
  which +
  (which === "translated" ? "?v=" + p.updated : "");
const best = (p) => url(p, p.ready ? "translated" : "original");
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
  localStorage.setItem("tralens-reader", JSON.stringify(prefs));
  document.documentElement.dataset.theme = prefs.theme;
}
document.documentElement.dataset.theme = prefs.theme;
function toast(message) {
  $("#notice").textContent = message;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => ($("#notice").textContent = ""), 4500);
}
async function api(path, body, reconnect = true) {
  const opts =
    body === undefined
      ? {}
      : { method: "POST", headers: { "X-Tralens-Token": config.token }, body };
  if (body !== undefined && !(body instanceof FormData)) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch("/api/" + path, opts);
  if (res.status === 401) {
    config = null;
    showOwnerLogin();
    throw Error("Sign in to your Tralens server.");
  }
  const data = await res.json();
  if (
    res.status === 403 &&
    reconnect &&
    body !== undefined &&
    /token|reload/i.test(data.error || "")
  ) {
    config = await api("config");
    return api(path, body, false);
  }
  if (!res.ok) throw Error(data.error || "Request failed.");
  return data;
}
async function refresh(force = false) {
  const data = await api("library");
  const changed = JSON.stringify(catalog) !== JSON.stringify(data);
  catalog = data;
  if (force || changed) render();
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
    best(pages[0]) +
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
      best(lastPage(recent)) +
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
function connectionButton() {
  return window.Capacitor?.isPluginAvailable?.("ReaderConnection")
    ? btn("configure-server", "Change server", "", "server") : "";
}

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
    (connectionButton() ? '<section class="panel fields"><h2>Server</h2><p>' + esc(location.origin) + '</p>' + connectionButton() + '</section>' : '') +
    (config.owner_mode ? btn("owner-logout", "Sign out of this server", "") : "") +
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
    best(pages[0]) +
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
          '</summary><div class="pages">' +
          c.pages
            .map(
              (p, i) =>
                '<a href="' +
                readURL(b, p) +
                '"><img loading="lazy" src="' +
                best(p) +
                '" alt="Page ' +
                (i + 1) +
                '">' +
                (i + 1) +
                "</a>",
            )
            .join("") +
          "</div></details>",
      )
      .join("") +
    "</div></main>"
  );
}
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
    '" loading="lazy">' +
    (reading ? '<button class="read-english" data-action="transcript" data-id="' + p.id + '">Read English</button>' : "") +
    (mode !== "original" && !p.ready
      ? '<small class="badge">Original</small>'
      : "") +
    "</div>"
  );
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
  const pending = ["queued", "running"].includes(p.status),
    error = p.error || (p.status === "skipped" ? "Translation skipped." : null);
  const banner =
    view !== "original" &&
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
        (pending || p.status === "waiting"
          ? btn("skip-page", "Skip page")
          : btn("retry-page", "Try again")) +
        ib("dismiss", "Dismiss", "x") +
        "</div>"
      : "";
  return (
    '<main class="reader ' +
    (prefs.background === "paper" ? "paper " : "") +
    (prefs.fit === "width" ? "width" : "") +
    '"><div class="canvas ' +
    (prefs.layout === "continuous" && view !== "compare" ? "continuous" : "") +
    '">' +
    content +
    "</div>" +
    banner +
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
let lastRender = "",
  lastFocusPage = "";
function render() {
  if (!config) return;
  const { screen, book, chapter, page } = current(),
    reading = screen === "read" && page;
  const searchFocused = document.activeElement?.id === "search",
    caret = searchFocused ? $("#search").selectionStart : 0;
  const wipeValue = $("#wipe")?.value;
  const continuous = $(".continuous"),
    oldScroll = continuous?.scrollTop,
    same = lastRender === location.hash;
  const hidden = same && $(".reader")?.classList.contains("clean");
  clearTimeout(hideTimer);
  root.innerHTML = reading
    ? reader(book, chapter, page)
    : screen === "settings"
      ? preferences()
      : screen === "book" && book
        ? detail(book)
        : library();
  document.title =
    (reading
      ? book.title + " · " + chapter.title
      : screen === "settings"
        ? "Settings"
        : book?.title || "Library") + " · Tralens";
  lastRender = location.hash;
  if (searchFocused && $("#search")) {
    $("#search").focus();
    $("#search").setSelectionRange(caret, caret);
  }
  if (reading) {
    if (same && wipeValue && $("#wipe")) $("#wipe").value = wipeValue;
    setWipe($("#wipe")?.value || 50);
    if (hidden) $(".reader").classList.add("clean");
    else showChrome();
    if ($(".continuous")) {
      const canvas = $(".continuous");
      const target = canvas.querySelector('[data-page="' + page.id + '"]');
      if (same && oldScroll !== undefined) canvas.scrollTop = oldScroll;
      else target?.scrollIntoView({ block: "start" });
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
            const next = chapter.pages.find((x) => x.id === el.dataset.page);
            history.replaceState(null, "", readURL(book, next));
            lastRender = location.hash;
            $("#scrub").value = chapter.pages.indexOf(next) + 1;
            $(".reader output").textContent =
              $("#scrub").value + " / " + chapter.pages.length;
            persistPosition(book, next);
            // Refresh chrome, page status and slider while retaining scroll offset.
            render();
          }
        }, 180);
      };
    }
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
function turn(delta, chapterOnly = false) {
  const { book, chapter, page } = current();
  if (!page) return;
  if (chapterOnly) {
    const c = book.chapters[book.chapters.indexOf(chapter) + delta];
    if (c) location.hash = readURL(book, c.pages[0]);
    return;
  }
  const pages = pagesOf(book),
    step =
      prefs.layout === "spread" && view !== "compare"
        ? delta > 0
          ? Math.min(2, chapter.pages.length - chapter.pages.indexOf(page))
          : chapter.pages.indexOf(page) === 1
            ? 1
            : 2
        : 1;
  const n =
    pages[
      Math.max(
        0,
        Math.min(pages.length - 1, pages.indexOf(page) + delta * step),
      )
    ];
  if (n) location.hash = readURL(book, n);
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
  if (uploadBusy) return;
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
sheet.addEventListener("cancel", (e) => {
  if (uploadBusy) e.preventDefault();
});
function addBook(id = "", files = []) {
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
}
function addFiles(files) {
  for (const file of [...files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { numeric: true }),
  )) {
    if (
      !/\.(png|jpe?g|webp|pdf|cbz|zip)$/i.test(file.name) ||
      (!/\.(pdf|cbz|zip)$/i.test(file.name) && file.size > config.max_upload_bytes)
    ) {
      toast("Choose PNG, JPEG, WebP, PDF, CBZ or ZIP. Images can be up to 32 MiB each.");
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
  modal(
    draft.book ? "Add chapter" : "Add manga",
    '<form id="import" class="fields"><div class="drop">' +
      icon("upload") +
      "<b>Drop manga pages here</b><small>PNG, JPEG, WebP, PDF, CBZ or ZIP · Up to 4 GiB per import</small>" +
      btn("choose", "Choose pages") +
      '<input id="files" type="file" accept="image/png,image/jpeg,image/webp,application/pdf,application/zip,application/x-zip-compressed,application/vnd.comicbook+zip,application/x-cbz,.cbz,.zip" multiple hidden></div>' +
      (draft.files.length
        ? "<div><small>" +
          draft.files.length +
          ' files · Reading order</small><div class="tray">' +
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
      '<p class="hint">' +
      esc(privacy()) +
      '</p><div class="error" role="alert" id="import-error"></div><button class="primary" ' +
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
        btn("download-book", "Download manga") +
        btn("retry-book", "Translate again") +
        btn("delete", "Delete from library", "danger") +
        "</div>",
    );
  if (kind === "download-book")
    return modal(
      "Download manga",
      '<div class="menu-list">' +
        ["translated", "original", "both"]
          .map(
            (v) =>
              '<a class="btn" href="/api/library/books/' +
              book.id +
              "/" +
              v +
              '.zip" download>' +
              esc(
                {
                  translated: "Available translated pages",
                  original: "Original pages",
                  both: "Originals and translations",
                }[v],
              ) +
              " · ZIP</a>",
          )
          .join("") +
        "</div>",
    );
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
  if (kind === "page-menu")
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
        btn("retry-page", "Translate again") +
        btn("fullscreen", "Toggle full screen") +
        "</div>",
    );
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
    return refresh(true);
  }
  if (action === "owner-logout") {
    const reply = await fetch("/api/owner/logout", {method: "POST"});
    if (reply.ok || reply.status === 401) location.reload();
    else toast("Could not sign out.");
    return;
  }
  if (action === "configure-server") return window.Capacitor.Plugins.ReaderConnection.configure();
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
  if (draft && ["title", "chapter"].includes(e.target.id))
    draft[e.target.id] = e.target.value;
});
function showChatGPTConsent(model) {
  modal("Connect to ChatGPT",
    '<p>Use your own ChatGPT plan on a Tralens server you control. The server stores your account ID, email and sign-in tokens. When you choose ChatGPT for an import or retry, Tralens sends the selected page images and detected text to OpenAI and processes the queue in the background.</p>' +
    '<p>You can disconnect in Tralens Settings. Read the <a href="/privacy" target="_blank" rel="noopener">Tralens privacy notice</a> and the <a href="https://openai.com/policies/privacy-policy/" target="_blank" rel="noopener">OpenAI Privacy Policy</a>.</p>' +
    '<button id="consent-chatgpt" class="primary" type="button">Agree &amp; continue</button>' +
    '<div id="chatgpt-connect" role="status"></div>');
  $("#consent-chatgpt").addEventListener("click", (event) => {
    event.currentTarget.remove();
    selectChatGPT(model, $("#chatgpt-connect"));
  }, {once: true});
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
          panel.textContent = "Finish signing in with ChatGPT in your phone’s browser, then return to Tralens.";
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
          ? '<p>Finish sign-in in the browser on your Tralens server. This reader will reconnect automatically.</p>' +
            '<details><summary>Using a headless server?</summary><p>Forward port ' + esc(url.port) +
            ' from your computer to the same loopback port on the server over SSH, then open this link on your computer.</p>' +
            '<a href="' + esc(url.href) + '" target="_blank" rel="noopener noreferrer">Continue with ChatGPT</a></details>'
          : '<p>Connect your ChatGPT account to use this model for manga translation.</p>' +
            '<a class="btn primary" href="' + esc(url.href) + '" target="_blank" rel="noopener noreferrer">Continue with ChatGPT</a>';
      }
      await new Promise(resolve => setTimeout(resolve, 800));
      if (!active()) return;
      result = await api("chatgpt/status");
      if (result.state === "connected") result = await api("chatgpt/connect", {});
    }
    if (!active()) return;
    if (result.state !== "connected") throw Error(result.error || "Sign-in did not finish.");
    config.chatgpt_connected = true;
    config.chatgpt_models = result.models;
    if (!result.models.includes(model)) throw Error("Your ChatGPT account does not offer this model.");
    prefs.translator = model;
    save();
    closeSheet();
    render();
  } catch (err) {
    if (active()) panel.textContent = err.message;
  }
}
document.addEventListener("change", (e) => {
  const el = e.target;
  if (el.dataset.pref) {
    if (el.dataset.pref === "translator" && el.value !== "local" && config.frontier_provider === "chatgpt") {
      const requested = el.value;
      el.value = prefs.translator;
      showChatGPTConsent(requested);
      return;
    }
    prefs[el.dataset.pref] = el.value;
    save();
    render();
  }
  if (el.id === "scrub") {
    const { book, chapter } = current();
    location.hash = readURL(book, chapter.pages[Number(el.value) - 1]);
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
document.addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = e.target.querySelector('button:not([type="button"])');
  if (button) button.disabled = true;
  try {
    if (e.target.id === "import") {
      uploadBusy = true;
      const form = new FormData();
      form.append(
        "metadata",
        JSON.stringify({
          book: draft.book,
          title: draft.title,
          chapter: draft.chapter,
          options: settings(),
        }),
      );
      draft.files.forEach((f) => form.append("page", f.file));
      await api("library/add", form);
      uploadBusy = false;
      closeSheet();
      await refresh(true);
      location.hash = "library";
      toast("Added to your library.");
    } else if (e.target.id === "rename") {
      await api("library/rename", {
        book: current().book.id,
        title: $("#new-title").value,
      });
      closeSheet();
      await refresh(true);
    }
  } catch (err) {
    if ($("#import-error")) $("#import-error").textContent = err.message;
    else toast(err.message);
  } finally {
    uploadBusy = false;
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
  if (uploadBusy) return;
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
addEventListener("hashchange", () => {
  if (!config) return;
  if (sheet.open) closeSheet();
  render();
  window.scrollTo(0, 0);
});
function showOwnerLogin() {
  if (root.querySelector("#owner-login")) return;
  if (sheet.open) sheet.close();
  catalog = {books: [], bytes: 0};
  root.innerHTML = '<main class="empty"><h1>Sign in to Tralens</h1><p>' + esc(location.origin) +
    '</p><form id="owner-login"><label>Server owner key <input type="password" required maxlength="256" autocomplete="current-password"></label> <button class="primary">Sign in</button><p role="alert"></p></form>' + connectionButton() + '</main>';
  root.querySelector("form").onsubmit = async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    try {
      const response = await fetch("/api/owner/login", {method: "POST", body: form.querySelector("input").value});
      if (!response.ok) throw Error((await response.json()).error);
      form.reset();
      await init();
    } catch (error) {
      form.querySelector('[role="alert"]').textContent = error.message;
    }
  };
}
async function init() {
  if (connectionButton()) root.innerHTML = '<main class="empty"><h1>Connecting to your library…</h1><p>' + esc(location.origin) + '</p>' + connectionButton() + '</main>';
  try {
    config = await api("config");
    if (prefs.translator !== "local" && (config.chatgpt_connected === false ||
        (config.chatgpt_models && !config.chatgpt_models.includes(prefs.translator)))) {
      prefs.translator = "local";
      save();
      toast("Local translation is selected. Choose a GPT model in Settings to connect.");
    }
    await refresh(true);
  } catch (err) {
    if (root.querySelector("#owner-login")) return;
    root.innerHTML =
      '<main class="empty"><h1>Library unavailable</h1><p>' +
      esc(err.message) +
      "</p>" +
      (connectionButton() ? '<p class="hint">Check that your phone can reach ' + esc(location.origin) + '.</p>' + connectionButton() : "") +
      btn("reconnect", "Reconnect", "primary") +
      "</main>";
  }
}
await init();
setInterval(async () => {
  if (!config || loading || document.hidden || uploadBusy) return;
  loading = true;
  try {
    await refresh();
  } catch (err) {
    toast(err.message);
  } finally {
    loading = false;
  }
}, 2500);

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
  if (area && !e.target.matches("input")) {
    area.setPointerCapture(e.pointerId);
    dragDivider(e);
  }
});
document.addEventListener("pointermove", (e) => {
  if (e.target.closest(".wipe")?.hasPointerCapture(e.pointerId)) dragDivider(e);
});
