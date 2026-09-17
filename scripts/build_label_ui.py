"""Generate the labelling console from the sampled golden set.

Two outputs from one template: an Artifact-shaped fragment (the tool wraps it in
a document at publish time) and a standalone file that opens from disk, so the
repo carries a runnable copy of exactly the instrument used to build the labels.

The page is deliberately given ONLY item_id and customer_text. The weak cluster
label, the region, and the brand's actual reply are all withheld: any of them
shown on screen would anchor the annotator, and the golden set would inherit the
clusterer's mistakes instead of checking them.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd   # noqa: E402

from hiver.schema import AUTO_REASONS, ESCALATION_REASONS, INTENTS  # noqa: E402

AUTO_KEYS = ["a", "s", "d"]
ESC_KEYS = ["z", "x", "c", "v", "b", "n", "m", ","]
INTENT_KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

TEMPLATE = r"""<title>__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --ground:#F6F8F9; --surface:#FFFFFF; --raise:#EEF2F4;
  --ink:#101619; --muted:#5A6B74; --line:#DCE3E7;
  --accent:#0E7C8B; --accent-soft:#DAEDF0;
  --auto:#14795A; --auto-soft:#DDF0E7;
  --esc:#A85A0B; --esc-soft:#FAEBDA;
  --flag:#A32F2F; --shadow:0 1px 2px rgba(16,22,25,.06),0 4px 14px rgba(16,22,25,.05);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0E1418; --surface:#161E23; --raise:#1D272D;
  --ink:#E4ECF0; --muted:#8798A2; --line:#243037;
  --accent:#3FB3C4; --accent-soft:#123038;
  --auto:#3CBE8C; --auto-soft:#10302561;
  --esc:#DE8C3E; --esc-soft:#33240F;
  --flag:#E06B6B; --shadow:0 1px 2px rgba(0,0,0,.4),0 6px 18px rgba(0,0,0,.3);
}}
:root[data-theme="dark"]{
  --ground:#0E1418; --surface:#161E23; --raise:#1D272D;
  --ink:#E4ECF0; --muted:#8798A2; --line:#243037;
  --accent:#3FB3C4; --accent-soft:#123038;
  --auto:#3CBE8C; --auto-soft:#103025;
  --esc:#DE8C3E; --esc-soft:#33240F;
  --flag:#E06B6B; --shadow:0 1px 2px rgba(0,0,0,.4),0 6px 18px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  margin:0;padding-block:0;line-height:1.5}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums}

header{position:sticky;top:0;z-index:10;background:var(--surface);
  border-bottom:1px solid var(--line)}
.bar{height:3px;background:var(--raise)}
.bar>div{height:100%;background:var(--accent);width:0;transition:width .18s ease}
.hrow{display:flex;align-items:center;gap:20px;flex-wrap:wrap;
  padding:10px 20px;max-width:900px;margin:0 auto}
.brand{font-weight:700;letter-spacing:-.01em;margin-right:auto;font-size:15px}
.brand span{color:var(--muted);font-weight:500}
.stat{display:flex;flex-direction:column;line-height:1.2}
.stat b{font-size:15px;font-weight:600}
.stat i{font-style:normal;font-size:10px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.08em}

main{max-width:900px;margin:0 auto;padding:16px 20px 90px}
.msg{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  box-shadow:var(--shadow);padding:18px 24px;margin-bottom:16px}
.msgmeta{display:flex;gap:10px;align-items:center;margin-bottom:10px}
.pill{font-size:10px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);border:1px solid var(--line);border-radius:99px;padding:2px 9px}
#prevnote{margin-top:14px;padding:9px 12px;border-left:2px solid var(--accent);
  background:var(--raise);border-radius:0 6px 6px 0;font-size:12.5px;color:var(--muted);
  max-width:62ch}
#prevnote b{color:var(--ink);font-weight:600}
.msgtext{font-size:19px;line-height:1.5;max-width:62ch;
  overflow-wrap:break-word;text-wrap:pretty}

.block{margin-bottom:14px}
.blabel{font-size:11px;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin-bottom:9px;display:flex;gap:10px;align-items:baseline}
.blabel em{font-style:normal;color:var(--ink);opacity:.55}
.grid{display:grid;gap:6px;grid-template-columns:repeat(auto-fill,minmax(215px,1fr))}
.opt{display:flex;gap:9px;align-items:flex-start;text-align:left;width:100%;
  background:var(--surface);border:1px solid var(--line);border-radius:7px;
  padding:7px 10px;cursor:pointer;color:inherit;font:inherit;
  transition:border-color .12s,background .12s}
.opt:hover{border-color:var(--accent)}
.opt:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.opt kbd{flex:none;width:19px;height:19px;display:grid;place-items:center;
  border:1px solid var(--line);border-radius:4px;background:var(--raise);
  font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:600;
  text-transform:uppercase;color:var(--muted)}
.opt .t{display:block;font-size:12.5px;font-weight:600;line-height:1.3}
.opt .d{display:block;font-size:10.5px;color:var(--muted);line-height:1.3;margin-top:2px}
.opt[aria-pressed="true"]{border-color:var(--accent);background:var(--accent-soft)}
.opt[aria-pressed="true"] kbd{background:var(--accent);color:var(--surface);
  border-color:var(--accent)}
.g-auto .opt[aria-pressed="true"]{border-color:var(--auto);background:var(--auto-soft)}
.g-auto .opt[aria-pressed="true"] kbd{background:var(--auto);border-color:var(--auto)}
.g-esc .opt[aria-pressed="true"]{border-color:var(--esc);background:var(--esc-soft)}
.g-esc .opt[aria-pressed="true"] kbd{background:var(--esc);border-color:var(--esc)}
.route{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:700px){.route{grid-template-columns:1fr}}
.rhead{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  margin-bottom:8px;display:flex;align-items:center;gap:7px}
.dot{width:7px;height:7px;border-radius:99px;flex:none}
.rhead.a{color:var(--auto)} .rhead.a .dot{background:var(--auto)}
.rhead.e{color:var(--esc)} .rhead.e .dot{background:var(--esc)}
.g-auto,.g-esc{grid-template-columns:1fr}

.extras{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-top:4px}
.mini{font:inherit;font-size:11.5px;color:var(--muted);background:var(--surface);
  border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
.mini:hover{border-color:var(--accent);color:var(--ink)}
.mini:disabled{opacity:.4;cursor:default}
.mini:disabled:hover{border-color:var(--line);color:var(--muted)}
.mini[aria-pressed="true"]{border-color:var(--flag);color:var(--flag);font-weight:600}
#note{flex:1;min-width:200px;font:inherit;font-size:12.5px;background:var(--surface);
  color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 10px}
#note:focus{outline:2px solid var(--accent);outline-offset:1px}

footer{position:fixed;bottom:0;left:0;right:0;background:var(--surface);
  border-top:1px solid var(--line);padding:8px 20px;font-size:11px;color:var(--muted)}
.frow{max-width:900px;margin:0 auto;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.frow b{color:var(--ink);font-weight:600}
.frow .sp{margin-left:auto}
kbd.lite{font-family:"IBM Plex Mono",monospace;border:1px solid var(--line);
  border-radius:3px;padding:0 4px;background:var(--raise)}
#save{font-size:10px;letter-spacing:.06em;text-transform:uppercase}
#save.ok{color:var(--auto)} #save.warn{color:var(--esc)} #save.err{color:var(--flag)}

#done{display:none;text-align:center;padding:70px 20px}
#done h2{font-size:26px;margin:0 0 8px}
#done p{color:var(--muted);max-width:52ch;margin:0 auto 20px}
#exp{width:100%;height:190px;font-family:"IBM Plex Mono",monospace;font-size:10.5px;
  background:var(--surface);color:var(--ink);border:1px solid var(--line);
  border-radius:8px;padding:11px;margin-top:16px}
.btn{font:inherit;font-size:13px;font-weight:600;background:var(--accent);
  color:#fff;border:0;border-radius:7px;padding:9px 18px;cursor:pointer}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<header>
  <div class="bar"><div id="prog"></div></div>
  <div class="hrow">
    <div class="brand">__TITLE__ <span>&middot; __SUB__</span></div>
    <div class="stat"><b id="s-done" class="mono">0</b><i>labelled</i></div>
    <div class="stat"><b id="s-left" class="mono">0</b><i>remaining</i></div>
    <div class="stat"><b id="s-rate" class="mono">&ndash;</b><i>per min</i></div>
    <div class="stat"><b id="s-eta" class="mono">&ndash;</b><i>est. left</i></div>
  </div>
</header>

<main>
  <div id="work">
    <div class="msg">
      <div class="msgmeta">
        <span class="pill mono" id="item-id">&mdash;</span>
        <span class="pill mono" id="item-n">&mdash;</span>
        <span class="pill" id="item-why" hidden></span>
        <span class="pill" id="item-seen" hidden>revisit</span>
      </div>
      <div class="msgtext" id="msg"></div>
      <div id="prevnote" hidden></div>
    </div>

    <div class="block">
      <div class="blabel">Intent <em>press 1&ndash;0</em></div>
      <div class="grid" id="g-intent"></div>
    </div>

    <div class="block">
      <div class="blabel">Routing decision <em>home row auto-handles &middot; bottom row escalates</em></div>
      <div class="route">
        <div>
          <div class="rhead a"><span class="dot"></span>Auto-handle</div>
          <div class="grid g-auto" id="g-auto"></div>
        </div>
        <div>
          <div class="rhead e"><span class="dot"></span>Escalate to human</div>
          <div class="grid g-esc" id="g-esc"></div>
        </div>
      </div>
    </div>

    <div class="extras">
      <button class="mini" id="flag" aria-pressed="false">Flag as hard &middot; F</button>
      <button class="mini" id="clear">Clear this item &middot; Esc</button>
      <input id="note" placeholder="Optional note &mdash; why this one was tricky">
      <button class="mini" id="back">&larr; Previous</button>
      <button class="mini" id="skip">Skip &rarr;</button>
    </div>
  </div>

  <div id="done">
    <h2>All 0 items labelled</h2>
    <p id="donesub"></p>
    <button class="btn" id="copy">Copy labels to clipboard</button>
    <button class="btn" id="dl" style="background:var(--raise);color:var(--ink)">Download JSON</button>
    <textarea id="exp" readonly class="mono"></textarea>
  </div>
</main>

<footer><div class="frow">
  <span><b>1&ndash;0</b> intent</span>
  <span><b>A S D</b> auto</span>
  <span><b>Z X C V B N M ,</b> escalate</span>
  <span><kbd class="lite">&#8592;</kbd> back</span>
  <span><kbd class="lite">Esc</kbd> clear</span>
  <span><kbd class="lite">F</kbd> flag</span>
  <span><kbd class="lite">/</kbd> note</span>
  <span class="sp mono" id="save">local only</span>
</div></footer>

<script>
const ITEMS = __ITEMS__;
const INTENTS = __INTENTS__;
const AUTO = __AUTO__;
const ESC = __ESC__;
const LS = "__LSKEY__";
const COLL = "__COLL__";

let labels = {};
let i = 0, shownAt = Date.now(), db = null;
try { labels = JSON.parse(localStorage.getItem(LS) || "{}"); } catch (e) { labels = {}; }

const $ = s => document.querySelector(s);
const cur = () => ITEMS[i];

function optionHTML(o, key) {
  return `<button class="opt" data-key="${key}" aria-pressed="false">
    <kbd>${key}</kbd><span><span class="t">${o.label}</span><span class="d">${o.desc}</span></span></button>`;
}
$("#g-intent").innerHTML = INTENTS.map(o => optionHTML(o, o.key)).join("");
$("#g-auto").innerHTML = AUTO.map(o => optionHTML(o, o.key)).join("");
$("#g-esc").innerHTML = ESC.map(o => optionHTML(o, o.key)).join("");

const keyToIntent = {}; INTENTS.forEach(o => keyToIntent[o.key] = o.name);
const keyToRoute = {};
AUTO.forEach(o => keyToRoute[o.key] = { route: "auto", reason: o.name });
ESC.forEach(o => keyToRoute[o.key] = { route: "escalate", reason: o.name });

function render(keepTime) {
  const it = cur();
  if (!it) return finish();
  $("#msg").textContent = it.text;
  $("#item-id").textContent = it.id;
  $("#item-n").textContent = (i + 1) + " / " + ITEMS.length;
  const rec = labels[it.id];
  $("#item-seen").hidden = !rec;
  const why = $("#item-why");
  why.hidden = !it.why; if (it.why) why.textContent = it.why.replace("_", " ");
  const pn = $("#prevnote");
  pn.hidden = !it.note;
  if (it.note) pn.innerHTML = "<b>Your note from the first pass:</b> " +
    it.note.replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));

  document.querySelectorAll(".opt").forEach(b => b.setAttribute("aria-pressed", "false"));
  if (rec) {
    const hit = INTENTS.find(o => o.name === rec.intent);
    if (hit) markKey("#g-intent", hit.key);
    const rk = Object.keys(keyToRoute).find(k => keyToRoute[k].reason === rec.reason);
    if (rk) markKey(rec.route === "auto" ? "#g-auto" : "#g-esc", rk);
  }
  $("#flag").setAttribute("aria-pressed", rec && rec.flagged ? "true" : "false");
  $("#note").value = (rec && rec.note) || "";
  $("#clear").disabled = !rec;
  if (!keepTime) shownAt = Date.now();
  stats();
}

function markKey(scope, key) {
  const b = document.querySelector(`${scope} .opt[data-key="${key}"]`);
  if (b) b.setAttribute("aria-pressed", "true");
}

function stats() {
  const done = Object.keys(labels).length, total = ITEMS.length;
  $("#s-done").textContent = done;
  $("#s-left").textContent = total - done;
  $("#prog").style.width = (done / total * 100) + "%";
  const times = Object.values(labels).map(r => r.ms).filter(Boolean);
  if (times.length >= 3) {
    const med = times.slice().sort((a, b) => a - b)[Math.floor(times.length / 2)];
    const perMin = 60000 / med;
    $("#s-rate").textContent = perMin.toFixed(1);
    const mins = Math.round((total - done) / perMin);
    $("#s-eta").textContent = mins >= 60 ? Math.floor(mins / 60) + "h " + (mins % 60) + "m" : mins + "m";
  }
}

function persist() {
  try { localStorage.setItem(LS, JSON.stringify(labels)); } catch (e) {}
}

let advanceTimer = null;
function cancelAdvance() { clearTimeout(advanceTimer); advanceTimer = null; }

function put(patch, opts) {
  opts = opts || {};
  const it = cur(); if (!it) return;
  const prev = labels[it.id] || {};
  const wasComplete = !!(prev.intent && prev.route);
  const rec = Object.assign({ item_id: it.id }, prev, patch);
  // An explicit null clears the field rather than storing null.
  Object.keys(patch).forEach(k => { if (patch[k] === null) delete rec[k]; });
  if (!prev.intent && patch.intent) rec.ms = Date.now() - shownAt;
  rec.ts = new Date().toISOString();
  labels[it.id] = rec;
  persist();
  save(rec);
  render(true);
  // Advance only when this keypress COMPLETED the item. Revisiting a finished
  // one to correct it must never whisk the annotator away mid-correction.
  if (!wasComplete && rec.intent && rec.route && !opts.noAdvance) {
    cancelAdvance();
    advanceTimer = setTimeout(() => {
      if (cur() && cur().id === rec.item_id) next();
    }, 420);
  }
}

function clearItem() {
  const it = cur(); if (!it) return;
  cancelAdvance();
  delete labels[it.id];
  persist();
  if (db) db.doc(COLL + "/" + it.id).delete().catch(() => {});
  render(true);
  stats();
}

let pending = 0;
function save(rec) {
  if (!db) return;
  pending++; setSave("saving…", "warn");
  db.doc(COLL + "/" + rec.item_id).set(rec)
    .then(() => { pending--; if (!pending) setSave("saved", "ok"); })
    .catch(() => { pending--; setSave("save failed — kept locally", "err"); });
}
function setSave(t, c) { const e = $("#save"); e.textContent = t; e.className = "mono " + c; }

function next() { i = Math.min(i + 1, ITEMS.length); render(); }
function prev() { i = Math.max(i - 1, 0); render(); }

function finish() {
  $("#work").style.display = "none";
  const d = $("#done"); d.style.display = "block";
  const n = Object.keys(labels).length;
  const esc = Object.values(labels).filter(r => r.route === "escalate").length;
  const fl = Object.values(labels).filter(r => r.flagged).length;
  d.querySelector("h2").textContent = "All " + n + " items labelled";
  $("#donesub").textContent = esc + " escalate, " + (n - esc) + " auto-handle, " + fl +
    " flagged as hard. Labels are saved; you can close this page.";
  $("#exp").value = JSON.stringify(Object.values(labels), null, 1);
}

document.addEventListener("click", e => {
  const b = e.target.closest(".opt"); if (!b) return;
  const key = b.dataset.key;
  if (b.closest("#g-intent")) apply(key); else apply(key);
});

function apply(key) {
  cancelAdvance();
  const it = cur(); if (!it) return;
  const rec = labels[it.id] || {};
  if (keyToIntent[key] && document.querySelector(`#g-intent .opt[data-key="${key}"]`)) {
    const same = rec.intent === keyToIntent[key];
    put({ intent: same ? null : keyToIntent[key] }, { noAdvance: same });
    return;
  }
  const r = keyToRoute[key];
  if (r) {
    const same = rec.route === r.route && rec.reason === r.reason;
    put(same ? { route: null, reason: null } : { route: r.route, reason: r.reason },
        { noAdvance: same });
  }
}

document.addEventListener("keydown", e => {
  if (e.target === $("#note")) { if (e.key === "Enter" || e.key === "Escape") { put({ note: $("#note").value }); $("#note").blur(); } return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "arrowleft" || k === "backspace") { e.preventDefault(); cancelAdvance(); prev(); return; }
  if (k === "arrowright") { e.preventDefault(); cancelAdvance(); next(); return; }
  if (e.key === "Escape") { e.preventDefault(); clearItem(); return; }
  if (k === "/") { e.preventDefault(); $("#note").focus(); return; }
  if (k === "f") { e.preventDefault(); const b = $("#flag"); const v = b.getAttribute("aria-pressed") !== "true";
    b.setAttribute("aria-pressed", v ? "true" : "false"); put({ flagged: v }); return; }
  if (keyToIntent[k] || keyToRoute[k]) { e.preventDefault(); apply(k); }
});

$("#flag").onclick = () => { const b = $("#flag"); const v = b.getAttribute("aria-pressed") !== "true";
  b.setAttribute("aria-pressed", v ? "true" : "false"); put({ flagged: v }); };
$("#clear").onclick = clearItem;
$("#back").onclick = () => { cancelAdvance(); prev(); };
$("#skip").onclick = () => { cancelAdvance(); next(); };
$("#note").onblur = () => put({ note: $("#note").value });
$("#copy").onclick = () => { navigator.clipboard.writeText($("#exp").value)
  .then(() => $("#copy").textContent = "Copied").catch(() => $("#exp").select()); };
$("#dl").onclick = async () => {
  const d = await (window.claude && window.claude.use ? window.claude.use("downloads") : null);
  if (!d) { $("#exp").select(); $("#dl").textContent = "Select the text and copy instead"; return; }
  try { await d.save({ filename: "golden_labels.json", data: $("#exp").value }); }
  catch (err) { $("#dl").textContent = "Download declined"; }
};

// Resume at the first unlabelled item.
const firstOpen = ITEMS.findIndex(it => !labels[it.id]);
i = firstOpen < 0 ? ITEMS.length : firstOpen;
render();

// db is optional: the page is fully usable on localStorage alone, and lights up
// server-side saving only if the capability resolves.
(async () => {
  if (!window.claude || !window.claude.use) return;
  db = await window.claude.use("db");
  if (!db) return;
  setSave("connected", "ok");
  try {
    const snap = await db.collection(COLL).get();
    let merged = 0;
    snap.docs.forEach(doc => {
      const d = doc.data ? doc.data() : doc;
      if (d && d.item_id && !labels[d.item_id]) { labels[d.item_id] = d; merged++; }
    });
    if (merged) {
      try { localStorage.setItem(LS, JSON.stringify(labels)); } catch (e) {}
      const f = ITEMS.findIndex(it => !labels[it.id]);
      i = f < 0 ? ITEMS.length : f;
      render();
    }
    stats();
  } catch (e) { setSave("offline — saving locally", "warn"); }
})();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", help="JSON list of items to review; omit for the full set")
    ap.add_argument("--collection", default="labels", help="artifact db collection")
    ap.add_argument("--ls-key", default="hiver_golden_v1")
    ap.add_argument("--title", default="Golden Set Console")
    ap.add_argument("--sub", default="SpotifyCares triage")
    ap.add_argument("--stem", default="label_ui")
    args = ap.parse_args()

    g = pd.read_parquet("data/golden/sample.parquet")
    text = dict(zip(g.item_id, g.customer_text))

    if args.queue:
        queue = json.loads(Path(args.queue).read_text())
        items = [{"id": q["id"], "text": text[q["id"]], "why": q["why"],
                  "note": q.get("note", "")} for q in queue]
    else:
        items = [{"id": r.item_id, "text": r.customer_text} for r in g.itertuples()]

    body = (TEMPLATE
            .replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
            .replace("__INTENTS__", json.dumps(
                [{"name": k, "key": INTENT_KEYS[n], "label": k.replace("_", " "), "desc": d}
                 for n, (k, d) in enumerate(INTENTS)]))
            .replace("__AUTO__", json.dumps(
                [{"name": k, "key": AUTO_KEYS[n], "label": k.replace("_", " "), "desc": d}
                 for n, (k, d) in enumerate(AUTO_REASONS)]))
            .replace("__ESC__", json.dumps(
                [{"name": k, "key": ESC_KEYS[n], "label": k.replace("_", " "), "desc": d}
                 for n, (k, d) in enumerate(ESCALATION_REASONS)]))
            .replace("__LSKEY__", args.ls_key)
            .replace("__COLL__", args.collection)
            .replace("__TITLE__", args.title)
            .replace("__SUB__", args.sub))

    assert len(INTENTS) <= len(INTENT_KEYS), "more intents than keys"
    assert len(ESCALATION_REASONS) <= len(ESC_KEYS), "more escalation reasons than keys"

    out = Path("tools")
    out.mkdir(exist_ok=True)
    (out / f"{args.stem}.artifact.html").write_text(body)
    (out / f"{args.stem}.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "</head><body>" + body + "</body></html>")

    print(f"{len(items)} items embedded -> tools/{args.stem}.artifact.html "
          f"({len(body)/1024:.0f} KB), db collection {args.collection!r}")

    # The annotator must never see a prior LABEL. Their own note is allowed
    # through (it is their reasoning, not an answer), so the check is on the
    # label-bearing fields specifically.
    embedded = json.loads(body.split("const ITEMS = ", 1)[1].split(";\n", 1)[0])
    keys = {k for it in embedded for k in it}
    banned = keys & {"intent", "route", "reason", "intent_weak", "region", "reply_text"}
    assert not banned, f"prior labels leaked into the UI: {banned}"
    leaked = [r.item_id for r in g.itertuples() if str(r.reply_text)[:45] in body]
    assert not leaked, f"reference replies leaked into the UI: {leaked[:3]}"
    print(f"  leak check: items expose {sorted(keys)}; no prior labels, no replies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
