// Only this worker can contact another origin. It never receives credentials.
self.onmessage = async ({data: origin}) => {
  try {
    const url = new URL(origin);
    if (url.origin !== origin || (url.protocol !== "https:" &&
        !(url.protocol === "http:" && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)))) throw Error();
    const response = await fetch(origin + "/api/auth/discovery", {
      credentials: "omit", redirect: "error", referrerPolicy: "no-referrer", cache: "no-store"
    });
    if (!response.ok || !response.headers.get("Content-Type")?.startsWith("application/json")) throw Error();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let bytes = 0, body = "";
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      bytes += value.length;
      if (bytes > 1024) throw Error();
      body += decoder.decode(value, {stream:true});
    }
    const info = JSON.parse(body + decoder.decode());
    self.postMessage(info.service === "tralens-reader" && info.version === 1);
  } catch { self.postMessage(false); }
};
