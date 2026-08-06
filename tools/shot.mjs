// Screenshot a URL with headless Chrome via CDP.
// Usage: node tools/shot.mjs <url> <out.png relative to dist/> [width] [height]
// Uses Emulation.setDeviceMetricsOverride so narrow (mobile) widths render at the
// true CSS width. Plain `--window-size` headless screenshots clamp the window to a
// 500px minimum width, which silently renders "390px" shots at a 500px layout
// cropped to 390 (fake horizontal overflow on every mobile shot).
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const [url, out, w = "1440", h = "3200"] = process.argv.slice(2);
if (!url || !out) {
  console.error("usage: node tools/shot.mjs <url> <out.png> [w] [h]");
  process.exit(1);
}
const width = parseInt(w, 10), height = parseInt(h, 10);
const outPath = resolve(join(root, "dist"), out);
mkdirSync(dirname(outPath), { recursive: true });

const chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const proc = spawn(chrome, [
  "--headless=new",
  "--disable-gpu",
  "--no-first-run",
  "--hide-scrollbars",
  "--remote-debugging-port=0",
  `--user-data-dir=/tmp/aether-shot-profile-${process.pid}`,
  "about:blank",
], { stdio: ["ignore", "pipe", "pipe"] });

const wsUrl = await new Promise((res, rej) => {
  let buf = "";
  const onData = (d) => {
    buf += d;
    const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
    if (m) res(m[1]);
  };
  proc.stderr.on("data", onData);
  proc.stdout.on("data", onData);
  proc.on("exit", () => rej(new Error("chrome exited early\n" + buf)));
  setTimeout(() => rej(new Error("timeout waiting for DevTools\n" + buf)), 20000);
});

const listRes = await fetch("http://" + wsUrl.split("//")[1].split("/")[0] + "/json/list");
const targets = await listRes.json();
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

let id = 0;
const pending = new Map();
const events = [];
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.id && pending.has(msg.id)) {
    const { res, rej } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result);
  } else if (msg.method) events.push(msg.method);
};
const send = (method, params = {}) =>
  new Promise((res, rej) => {
    const mid = ++id;
    pending.set(mid, { res, rej });
    ws.send(JSON.stringify({ id: mid, method, params }));
  });

await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", {
  width, height, deviceScaleFactor: 1, mobile: width < 800,
});
await send("Page.navigate", { url });
await new Promise((res) => {
  const t0 = Date.now();
  const tick = setInterval(() => {
    if (events.includes("Page.loadEventFired") || Date.now() - t0 > 15000) {
      clearInterval(tick); res();
    }
  }, 50);
});
await new Promise((r) => setTimeout(r, 400));

// Capture at least the requested height, but never cut the page off mid-flow:
// extend the viewport to the full content height so reveal-on-scroll fires,
// then capture everything.
const evalH = await send("Runtime.evaluate", {
  expression: "document.documentElement.scrollHeight", returnByValue: true,
});
const fullH = Math.max(height, Math.ceil(evalH.result.value || height));
if (fullH > height) {
  await send("Emulation.setDeviceMetricsOverride", {
    width, height: fullH, deviceScaleFactor: 1, mobile: width < 800,
  });
}
await new Promise((r) => setTimeout(r, 900)); // fonts + reveal transitions

const shot = await send("Page.captureScreenshot", {
  format: "png",
  clip: { x: 0, y: 0, width, height: fullH, scale: 1 },
  captureBeyondViewport: true,
});
writeFileSync(outPath, Buffer.from(shot.data, "base64"));
ws.close();
proc.kill();
console.log(`shot ${url} → ${outPath} (${width}x${height})`);
