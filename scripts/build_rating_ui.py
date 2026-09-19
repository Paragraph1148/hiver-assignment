"""Build the human rating queue that validates the LLM judge.

An LLM judge is only worth its scores if it agrees with a person. This samples
(item, system) pairs, strips every clue about which system wrote which reply,
and asks a human for the same four rubric scores the judge gives. Agreement
between the two becomes the reported ceiling on every judged number.

Blinding matters more here than in intent labelling. A rater who knows a reply
is "the system" rather than "the baseline" will score it differently, and the
agreement figure would then measure the rater's expectations rather than the
judge's accuracy. Replies are shuffled within each item and the system name is
never sent to the page.

The rater sees the same precedents the judge sees, so neither is working from
more context than the other.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from hiver.judge import DIMENSIONS  # noqa: E402

SYSTEMS = ["draft_canned", "draft_nearest", "draft_rag", "reply_text"]

TEMPLATE = r"""<title>Reply Rating Bench</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{--ground:#F6F8F9;--surface:#FFFFFF;--raise:#EEF2F4;--ink:#101619;--muted:#5A6B74;
  --line:#DCE3E7;--accent:#0E7C8B;--accent-soft:#DAEDF0;--good:#14795A;--bad:#A85A0B;
  --shadow:0 1px 2px rgba(16,22,25,.06),0 4px 14px rgba(16,22,25,.05)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0E1418;--surface:#161E23;--raise:#1D272D;--ink:#E4ECF0;--muted:#8798A2;
  --line:#243037;--accent:#3FB3C4;--accent-soft:#123038;--good:#3CBE8C;--bad:#DE8C3E;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 6px 18px rgba(0,0,0,.3)}}
:root[data-theme="dark"]{--ground:#0E1418;--surface:#161E23;--raise:#1D272D;--ink:#E4ECF0;
  --muted:#8798A2;--line:#243037;--accent:#3FB3C4;--accent-soft:#123038;--good:#3CBE8C;
  --bad:#DE8C3E;--shadow:0 1px 2px rgba(0,0,0,.4),0 6px 18px rgba(0,0,0,.3)}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;line-height:1.5;
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif}
.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
header{position:sticky;top:0;z-index:10;background:var(--surface);border-bottom:1px solid var(--line)}
.bar{height:3px;background:var(--raise)}.bar>div{height:100%;background:var(--accent);width:0;transition:width .18s}
.hrow{display:flex;align-items:center;gap:20px;flex-wrap:wrap;padding:10px 20px;max-width:880px;margin:0 auto}
.brand{font-weight:700;font-size:15px;margin-right:auto}.brand span{color:var(--muted);font-weight:500}
.stat{display:flex;flex-direction:column;line-height:1.2}
.stat b{font-size:15px;font-weight:600}
.stat i{font-style:normal;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
main{max-width:880px;margin:0 auto;padding:16px 20px 90px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  box-shadow:var(--shadow);padding:16px 22px;margin-bottom:14px}
.lbl{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:6px}
.cust{font-size:17px;line-height:1.5;max-width:62ch;overflow-wrap:break-word}
.reply{font-size:17px;line-height:1.55;max-width:62ch;border-left:3px solid var(--accent);
  padding-left:14px;overflow-wrap:break-word}
details{margin-top:12px}
details summary{cursor:pointer;font-size:11.5px;color:var(--muted)}
details ul{margin:8px 0 0;padding-left:18px;font-size:12px;color:var(--muted)}
details li{margin-bottom:5px;line-height:1.4}
.dim{display:grid;grid-template-columns:180px 1fr;gap:12px;align-items:center;
  padding:9px 0;border-top:1px solid var(--line)}
.dim:first-of-type{border-top:0}
.dim .n{font-size:13px;font-weight:600}
.dim .d{font-size:11px;color:var(--muted);line-height:1.35}
.scale{display:flex;gap:6px}
.scale button{flex:1;font:inherit;font-size:13px;font-weight:600;padding:7px 0;cursor:pointer;
  background:var(--surface);color:var(--muted);border:1px solid var(--line);border-radius:6px;
  transition:background .12s,border-color .12s,color .12s}
.scale button:hover{border-color:var(--accent);color:var(--ink)}
.scale button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.scale button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
@media(max-width:620px){.dim{grid-template-columns:1fr}}
footer{position:fixed;bottom:0;left:0;right:0;background:var(--surface);border-top:1px solid var(--line);
  padding:8px 20px;font-size:11px;color:var(--muted)}
.frow{max-width:880px;margin:0 auto;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.frow b{color:var(--ink);font-weight:600}
kbd{font-family:"IBM Plex Mono",monospace;border:1px solid var(--line);border-radius:3px;
  padding:0 4px;background:var(--raise)}
#save{margin-left:auto;font-size:10px;text-transform:uppercase;letter-spacing:.06em}
#save.ok{color:var(--good)}#save.warn{color:var(--bad)}
.mini{font:inherit;font-size:11.5px;color:var(--muted);background:var(--surface);
  border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
#done{display:none;text-align:center;padding:60px 20px}
#done h2{font-size:24px;margin:0 0 8px}
#done p{color:var(--muted);max-width:52ch;margin:0 auto}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<header>
  <div class="bar"><div id="prog"></div></div>
  <div class="hrow">
    <div class="brand">Reply Rating Bench <span>&middot; judge validation</span></div>
    <div class="stat"><b id="s-done" class="mono">0</b><i>rated</i></div>
    <div class="stat"><b id="s-left" class="mono">0</b><i>left</i></div>
    <div class="stat"><b id="s-eta" class="mono">&ndash;</b><i>est. left</i></div>
  </div>
</header>
<main>
  <div id="work">
    <div class="card">
      <div class="lbl mono" id="pos">&mdash;</div>
      <div class="cust" id="cust"></div>
    </div>
    <div class="card">
      <div class="lbl">Draft reply &mdash; would you send this?</div>
      <div class="reply" id="reply"></div>
      <details><summary>Show how Spotify handled similar cases</summary><ul id="prec"></ul></details>
    </div>
    <div class="card" id="dims"></div>
    <div style="display:flex;gap:9px"><button class="mini" id="back">&larr; Previous</button>
      <button class="mini" id="skip">Skip &rarr;</button></div>
  </div>
  <div id="done"><h2>All replies rated</h2><p id="donesub"></p></div>
</main>
<footer><div class="frow">
  <span><b>1&ndash;5</b> score each row, top to bottom</span>
  <span><kbd>&#8592;</kbd> back</span>
  <span>1 = unacceptable &middot; 3 = fine to send &middot; 5 = excellent</span>
  <span class="mono" id="save">local only</span>
</div></footer>
<script>
const ITEMS = __ITEMS__, DIMS = __DIMS__, LS = "hiver_ratings_v1";
let ratings = {}, i = 0, shownAt = Date.now(), db = null, focusDim = 0;
try { ratings = JSON.parse(localStorage.getItem(LS) || "{}"); } catch (e) {}
const $ = s => document.querySelector(s);
const cur = () => ITEMS[i];

$("#dims").innerHTML = DIMS.map((d, n) => `<div class="dim" data-dim="${n}">
  <div><div class="n">${d.name}</div><div class="d">${d.desc}</div></div>
  <div class="scale">${[1,2,3,4,5].map(v =>
    `<button data-dim="${n}" data-v="${v}" aria-pressed="false">${v}</button>`).join("")}</div></div>`).join("");

function render(keepTime) {
  const it = cur(); if (!it) return finish();
  $("#cust").textContent = it.customer;
  $("#reply").textContent = it.reply;
  $("#pos").textContent = (i + 1) + " / " + ITEMS.length;
  $("#prec").innerHTML = (it.precedents || []).map(p => `<li>${esc(p)}</li>`).join("")
    || "<li>No close precedents found.</li>";
  const r = ratings[it.key] || {};
  document.querySelectorAll(".scale button").forEach(b =>
    b.setAttribute("aria-pressed", String(r[DIMS[+b.dataset.dim].key] == b.dataset.v)));
  focusDim = DIMS.findIndex(d => !(d.key in r));
  if (focusDim < 0) focusDim = DIMS.length - 1;
  if (!keepTime) shownAt = Date.now();
  stats();
}
function esc(s){return String(s).replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));}

function stats() {
  const done = Object.values(ratings).filter(r => DIMS.every(d => d.key in r)).length;
  $("#s-done").textContent = done; $("#s-left").textContent = ITEMS.length - done;
  $("#prog").style.width = (done / ITEMS.length * 100) + "%";
  const t = Object.values(ratings).map(r => r._ms).filter(Boolean);
  if (t.length >= 3) {
    const med = t.slice().sort((a,b)=>a-b)[Math.floor(t.length/2)];
    const mins = Math.round((ITEMS.length - done) * med / 60000);
    $("#s-eta").textContent = mins >= 60 ? Math.floor(mins/60)+"h "+(mins%60)+"m" : mins+"m";
  }
}

function setScore(dimIdx, v) {
  const it = cur(); if (!it) return;
  const r = ratings[it.key] || { key: it.key, item_id: it.item_id, system: it.system };
  r[DIMS[dimIdx].key] = v;
  if (DIMS.every(d => d.key in r) && !r._ms) r._ms = Date.now() - shownAt;
  r._ts = new Date().toISOString();
  ratings[it.key] = r;
  try { localStorage.setItem(LS, JSON.stringify(ratings)); } catch (e) {}
  save(r);
  document.querySelectorAll(`.scale button[data-dim="${dimIdx}"]`).forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.v == v)));
  if (DIMS.every(d => d.key in r)) setTimeout(() => { if (cur() && cur().key === r.key) next(); }, 280);
  else focusDim = DIMS.findIndex(d => !(d.key in r));
  stats();
}
let pending = 0;
function save(r) {
  if (!db) return;
  pending++; $("#save").textContent = "saving…"; $("#save").className = "mono warn";
  db.doc("ratings/" + r.key).set(r).then(() => { if (!--pending) {
    $("#save").textContent = "saved"; $("#save").className = "mono ok"; } })
    .catch(() => { pending--; $("#save").textContent = "kept locally"; });
}
function next(){ i = Math.min(i+1, ITEMS.length); render(); }
function prev(){ i = Math.max(i-1, 0); render(); }
function finish(){
  $("#work").style.display="none"; $("#done").style.display="block";
  $("#donesub").textContent = Object.keys(ratings).length +
    " replies rated. These validate the automated judge — you can close this page.";
}
document.addEventListener("click", e => {
  const b = e.target.closest(".scale button"); if (b) setScore(+b.dataset.dim, +b.dataset.v);
});
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); prev(); return; }
  if (e.key === "ArrowRight") { e.preventDefault(); next(); return; }
  if (/^[1-5]$/.test(e.key)) { e.preventDefault(); setScore(focusDim, +e.key); }
});
$("#back").onclick = prev; $("#skip").onclick = next;
const first = ITEMS.findIndex(it => !(ratings[it.key] && DIMS.every(d => d.key in ratings[it.key])));
i = first < 0 ? ITEMS.length : first;
render();
(async () => {
  if (!window.claude || !window.claude.use) return;
  db = await window.claude.use("db"); if (!db) return;
  $("#save").textContent = "connected"; $("#save").className = "mono ok";
  try {
    const snap = await db.collection("ratings").get();
    let merged = 0;
    snap.docs.forEach(doc => { const d = doc.data ? doc.data() : doc;
      if (d && d.key && !ratings[d.key]) { ratings[d.key] = d; merged++; } });
    if (merged) { try { localStorage.setItem(LS, JSON.stringify(ratings)); } catch(e){}
      const f = ITEMS.findIndex(it => !(ratings[it.key] && DIMS.every(d => d.key in ratings[it.key])));
      i = f < 0 ? ITEMS.length : f; render(); }
    stats();
  } catch (e) { $("#save").textContent = "offline — saving locally"; }
})();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="(item, system) pairs to rate")
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    d = pd.read_parquet("data/results/_part_draft.parquet")
    rng = np.random.default_rng(args.seed)

    per = args.n // len(SYSTEMS)
    rows, key_map = [], {}
    for sysname in SYSTEMS:
        pick = rng.choice(len(d), size=min(per, len(d)), replace=False)
        for idx in pick:
            r = d.iloc[idx]
            # Opaque key. A key like "g0042:draft_rag" would unblind the rating
            # to anyone who opened devtools, and the system name must not be in
            # the page at all - the mapping stays here, keyed by a hash.
            key = hashlib.sha256(
                f"{args.seed}:{r.item_id}:{sysname}".encode()).hexdigest()[:16]
            key_map[key] = {"item_id": r.item_id, "system": sysname}
            rows.append({"key": key, "customer": r.customer_text,
                         "reply": str(getattr(r, sysname)),
                         "precedents": [p for p in str(r.precedents).split(" ||| ") if p][:3]})

    # Shuffle so consecutive items are not from the same system - a rater who
    # spots a run of identical canned replies starts scoring the pattern.
    rng.shuffle(rows)

    body = (TEMPLATE
            .replace("__ITEMS__", json.dumps(rows, ensure_ascii=False))
            .replace("__DIMS__", json.dumps(
                [{"key": k, "name": k, "desc": v} for k, v in DIMENSIONS])))

    Path("tools").mkdir(exist_ok=True)
    Path("tools/rating_ui.artifact.html").write_text(body)
    Path("tools/rating_ui.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "</head><body>" + body + "</body></html>")

    Path("data/golden/rating_key_map.json").write_text(json.dumps(
        {"seed": args.seed, "map": key_map}, indent=2))

    # The blinding is asserted, not assumed: nothing in the page may name a
    # system or carry a field that identifies one.
    embedded = json.loads(body.split("const ITEMS = ", 1)[1].split(", DIMS =", 1)[0])
    keys = {k for r in embedded for k in r}
    assert keys == {"key", "customer", "reply", "precedents"}, \
        f"page payload carries extra fields: {keys - {'key','customer','reply','precedents'}}"
    for name in SYSTEMS:
        assert name not in body, f"system name {name!r} leaked into the page"
    print(f"{len(rows)} (item, system) pairs, {per} per system, order shuffled")
    print(f"  page payload exposes only {sorted(keys)}")
    print("  blinding asserted: no system name appears anywhere in the page")
    print("  key -> system map kept locally in data/golden/rating_key_map.json")
    print("wrote tools/rating_ui.artifact.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
