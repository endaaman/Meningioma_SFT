"""patho2_gliomas_review.py — 院内グリオーマ（Patho2Gliomas）の 1 症例 1 枚の代表 HE を目で選ぶツール。

1 症例に HE が複数あるとき全部使うと腫瘍の写っていない切片を含めてしまうので、1 症例 1 枚にする（ken 2026-10-25）。
代表を決めるのは ken のクリックだけ（自動の規則は無い）。HE が 1 枚しかない症例は最初から選択済み（除外はできる）。
結果は data/patho2_gliomas/representatives.csv（case, file, note）の 1 ファイルで、解析はこれを読む。
file が EXCLUDE の症例は除外。note は single（HE 1 枚）/ manual（クリック）/ exclude。

手順:
    thumbs   slides.csv（patho2_gliomas_extract.py が書く）の HE 全部について WSI からサムネイル（長辺 thumb_px、JPEG）を作る
             → out/patho2_gliomas/review/thumbs/<群>/<症例>/<元名>.jpg。既存はスキップ
    serve    ブラウザで選ぶサーバーを起動（{review_host}:{review_port}。SSH 先のブラウザから開くなら review_host を 0.0.0.0 に）。
             サムネイルをダブルクリックすると WSI をそのまま拡大して見られる（wsi_toolbox.dzi で NDPI から Deep Zoom タイルをその場で切り、
             OpenSeadragon（cdnjs）で表示。事前生成なし）。クリックした瞬間に representatives.csv が書かれる。
             起動時に HE 1 枚の症例を選択済みとして埋める。thumbs は先に済ませておく
    html     サーバーを使わないとき用: 静的な out/patho2_gliomas/review/index.html を組む（選択はブラウザに保存、「CSV を書き出す」で
             representatives.csv を保存して data/patho2_gliomas/ に置く）

Usage:
    uv run python -u patho2_gliomas_review.py thumbs
    uv run python -u patho2_gliomas_review.py serve
    uv run python -u patho2_gliomas_review.py html
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from utils.loader import load_config

GROUP_NAMES = {"A": "A: IDH 変異型星細胞腫", "G": "G: IDH 野生型", "O": "O: 乏突起膠腫", "L": "L: リンパ腫", "M": "M: 転移"}
CSV_FIELDS = ["case", "file", "note"]


def review_dir(cfg: dict) -> Path:
    return Path(cfg["output_dir"]) / "patho2_gliomas" / "review"


def representatives_path(cfg: dict) -> Path:
    return Path(cfg["patho2_gliomas"].get("representatives", "data/patho2_gliomas/representatives.csv"))


def thumb_path(rdir: Path, r) -> Path:
    return rdir / "thumbs" / r.group / str(r.case) / (Path(r.file).stem + ".jpg")


# ---- thumbs ----

def make_thumb(wsi: str, out: Path, long_px: int) -> str:
    import openslide
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jpg.tmp")
    try:
        s = openslide.OpenSlide(wsi)
        img = s.get_thumbnail((long_px, long_px)).convert("RGB")
        img.save(tmp, "JPEG", quality=85)
        tmp.rename(out)
        return "ok"
    except Exception as e:  # 壊れた WSI などは飛ばす
        tmp.unlink(missing_ok=True)
        return f"error: {type(e).__name__}: {str(e)[:60]}"


def cmd_thumbs(cfg: dict, workers: int) -> None:
    rdir = review_dir(cfg)
    slides = pd.read_csv(Path(cfg["output_dir"]) / "patho2_gliomas" / "slides.csv")
    he = slides[slides["is_he"]]
    long_px = int(cfg["patho2_gliomas"].get("thumb_px", 1400))
    todo = [(r.wsi, thumb_path(rdir, r)) for r in he.itertuples(index=False) if not thumb_path(rdir, r).exists()]
    print(f"HE {len(he)}, thumbnails to make {len(todo)} (long edge {long_px} px, {workers} workers) -> {rdir / 'thumbs'}")
    t0 = time.time()
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(make_thumb, wsi, out, long_px): (wsi, out) for wsi, out in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            st = fut.result()
            counts[st.split(":")[0]] = counts.get(st.split(":")[0], 0) + 1
            if st != "ok" or i % 25 == 0:
                print(f"  [{i}/{len(todo)}] {st}  {futs[fut][1].relative_to(rdir)}  ({(time.time() - t0) / 60:.1f} min)")
    print(f"done in {(time.time() - t0) / 60:.1f} min: {counts}")


# ---- 症例と選択 ----

def load_cases(cfg: dict) -> list[dict]:
    """slides.csv の HE を症例ごとにまとめる（サムネイルのパス・パッチ数・再切片つき）。"""
    out_dir = Path(cfg["output_dir"]) / "patho2_gliomas"
    rdir = review_dir(cfg)
    slides = pd.read_csv(out_dir / "slides.csv")
    he = slides[slides["is_he"]].copy()
    if "n_patches" not in he:          # --dry-run の index には抽出の列が無い
        he["n_patches"] = float("nan")
    he["n_patches"] = pd.to_numeric(he["n_patches"], errors="coerce")
    order = list(cfg["patho2_gliomas"].get("priority", ["A", "G", "O", "L", "M"]))
    cases = []
    slide_paths: list[str] = []   # id → WSI のパス（serve の DZI 用）
    for (group, case), df in he.groupby(["group", "case"], sort=False):
        items = []
        for r in df.sort_values("file").itertuples(index=False):
            tp = thumb_path(rdir, r)
            items.append({
                "id": len(slide_paths), "file": r.file, "recut": bool(r.recut),
                "n_patches": None if pd.isna(r.n_patches) else int(r.n_patches),
                "thumb": str(tp.relative_to(rdir)) if tp.exists() else None,
            })
            slide_paths.append(r.wsi)   # id = append 前の長さ
        cases.append({"group": group, "label": df["label"].iloc[0], "case": str(case), "slides": items})
    cases.sort(key=lambda c: (order.index(c["group"]) if c["group"] in order else 99, c["case"]))
    return cases, slide_paths


def read_choices(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    return {r.case: {"file": r.file, "note": r.note} for r in df.itertuples(index=False) if r.case}


def write_choices(path: Path, choices: dict[str, dict], cases: list[dict]) -> None:
    """症例の並び（群 → 症例番号）で書く。.tmp に書いてから置き換える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for c in cases:
            ch = choices.get(c["case"])
            if ch:
                w.writerow({"case": c["case"], "file": ch["file"], "note": ch.get("note", "")})
    tmp.replace(path)


def prefill_single(choices: dict[str, dict], cases: list[dict]) -> int:
    """HE が 1 枚だけの症例を選択済みにする（既に何か入っていれば触らない）。"""
    n = 0
    for c in cases:
        if len(c["slides"]) == 1 and c["case"] not in choices:
            choices[c["case"]] = {"file": c["slides"][0]["file"], "note": "single"}
            n += 1
    return n


def page_data(cfg: dict, cases: list[dict]) -> dict:
    order = list(cfg["patho2_gliomas"].get("priority", ["A", "G", "O", "L", "M"]))
    return {"cases": cases, "groups": {g: GROUP_NAMES.get(g, g) for g in order},
            "csv": str(representatives_path(cfg)), "generated": time.strftime("%Y-%m-%d %H:%M")}


def build_html(cfg: dict, cases: list[dict]) -> str:
    data_json = json.dumps(page_data(cfg, cases), ensure_ascii=False)
    return HTML_TEMPLATE.replace("__DATA__", data_json.replace("</", "<\\/"))


def cmd_html(cfg: dict) -> None:
    rdir = review_dir(cfg)
    cases, _ = load_cases(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "index.html").write_text(build_html(cfg, cases), encoding="utf-8")
    n_thumb = sum(1 for c in cases for s in c["slides"] if s["thumb"])
    print(f"-> {rdir / 'index.html'}: {len(cases)} cases, {sum(len(c['slides']) for c in cases)} HE (thumbnails {n_thumb})")


# ---- serve ----

def _shown_host(host: str) -> str:
    """0.0.0.0 で待つときは案内用にこのホストの IP を出す。"""
    if host not in ("0.0.0.0", "::"):
        return host
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("10.255.255.255", 1)); ip = s.getsockname()[0]; s.close()
        return ip
    except OSError:
        return socket.gethostname()


class DziCache:
    """WSI を開いたまま持ち、Deep Zoom のタイルをその場で切る（スライドごとにロック。開くのは最大 max_open 本）。"""

    def __init__(self, slide_paths: list[str], max_open: int = 8, tile_size: int = 256):
        self.paths, self.max_open, self.tile_size = slide_paths, max_open, tile_size
        self.lock = threading.Lock()
        self.open: dict[int, tuple[threading.Lock, object]] = {}   # id → (lock, DziGenerator)
        self.order: list[int] = []

    def get(self, sid: int):
        from wsi_toolbox import DziGenerator, create_wsi_file
        with self.lock:
            if sid in self.open:
                self.order.remove(sid); self.order.append(sid)
                return self.open[sid]
            if sid < 0 or sid >= len(self.paths):
                raise KeyError(sid)
            while len(self.order) >= self.max_open:
                old = self.order.pop(0); self.open.pop(old, None)
            entry = (threading.Lock(), DziGenerator(create_wsi_file(self.paths[sid]), tile_size=self.tile_size))
            self.open[sid] = entry; self.order.append(sid)
            return entry

    def xml(self, sid: int) -> bytes:
        lk, dzi = self.get(sid)
        with lk:
            return dzi.xml().encode("utf-8")

    def tile(self, sid: int, level: int, col: int, row: int) -> bytes | None:
        from wsi_toolbox import DziTileNotFound, encode_tile
        lk, dzi = self.get(sid)
        with lk:
            try:
                return encode_tile(dzi.tile(level, col, row), quality=85)
            except DziTileNotFound:
                return None


DZI_RE = re.compile(r"^/dzi/(\d+)\.dzi$")
TILE_RE = re.compile(r"^/dzi/(\d+)_files/(\d+)/(\d+)_(\d+)\.jpe?g$")


def cmd_serve(cfg: dict, host: str, port: int) -> None:
    rdir = review_dir(cfg)
    csv_path = representatives_path(cfg)
    cases, slide_paths = load_cases(cfg)
    lock = threading.Lock()
    tiles = DziCache(slide_paths)
    choices = read_choices(csv_path)
    n_single = prefill_single(choices, cases)
    write_choices(csv_path, choices, cases)
    html_bytes = build_html(cfg, cases).encode("utf-8")
    print(f"cases {len(cases)}, HE {sum(len(c['slides']) for c in cases)}, "
          f"選択済み {len(choices)}（HE 1 枚で自動 {n_single}）\ncsv: {csv_path}\nlisten: {host}:{port}  open: http://{_shown_host(host)}:{port}/")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(rdir), **kw)

        def log_message(self, fmt, *args):  # タイルや静的ファイルのログは出さない
            if self.command == "POST" or "error" in fmt:
                super().log_message(fmt, *args)

        def _json(self, obj, code: int = 200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
            elif self.path == "/state":
                with lock:
                    self._json({"choices": choices, "csv": str(csv_path)})
            elif (m := DZI_RE.match(self.path)):
                self._bytes(tiles.xml(int(m.group(1))), "application/xml")
            elif (m := TILE_RE.match(self.path)):
                sid, level, col, row = (int(x) for x in m.groups())
                try:
                    body = tiles.tile(sid, level, col, row)
                except Exception as e:  # 開けない WSI など
                    self.log_message("dzi error %s: %s", self.path, e); body = None
                if body is None:
                    self.send_error(404)
                else:
                    self._bytes(body, "image/jpeg")
            else:
                super().do_GET()

        def _bytes(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/choice":
                self._json({"error": "not found"}, 404)
                return
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            case, file = str(req.get("case", "")), req.get("file")
            with lock:
                if not any(c["case"] == case for c in cases):
                    self._json({"error": f"unknown case {case}"}, 400)
                    return
                if file is None:
                    choices.pop(case, None)
                else:
                    choices[case] = {"file": file, "note": "exclude" if file == "EXCLUDE" else "manual"}
                write_choices(csv_path, choices, cases)
                self._json({"choices": choices, "csv": str(csv_path)})

    ThreadingHTTPServer((host, port), Handler).serve_forever()


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>Patho2Gliomas 代表 HE の選択</title>
<style>
:root{--sel:#1769aa;--ex:#b23b3b;--bg:#fafafa;--card:#fff;--line:#ddd;--muted:#777}
body{font-family:system-ui,-apple-system,"Noto Sans JP",sans-serif;margin:0;background:var(--bg);color:#222}
header{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header b{font-size:15px}
button,label.btn{font:inherit;padding:5px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
button.primary{background:var(--sel);color:#fff;border-color:var(--sel)}
.filters{display:flex;gap:6px;align-items:center;margin-left:auto}
.filters select{font:inherit;padding:4px}
main{padding:12px 16px}
.case{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:12px 0;padding:10px 12px}
.case.done{border-left:6px solid var(--sel)}
.case.excluded{border-left:6px solid var(--ex);opacity:.75}
.case h3{margin:0 0 6px;font-size:15px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.case h3 .meta{color:var(--muted);font-weight:normal;font-size:13px}
.case h3 .state{margin-left:auto;font-size:13px;font-weight:normal}
.slides{display:flex;gap:10px;flex-wrap:wrap}
.slide{width:300px;border:3px solid transparent;border-radius:8px;padding:4px;background:#f3f3f3}
.slide .imgwrap{cursor:zoom-in}
.slide .imgwrap:hover img{outline:2px solid #9bbce0}
.selbtn{font-size:12px;padding:2px 8px;margin:4px 6px 4px 0;border:1px solid #9bbce0;border-radius:6px;background:#fff;cursor:pointer}
.selbtn.on{background:var(--sel);color:#fff;border-color:var(--sel)}
.slide.selected{border-color:var(--sel);background:#e3eef9}
.slide img{width:100%;height:auto;display:block;border-radius:4px;background:#fff}
.slide .noimg{height:200px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px}
.slide .cap{font-size:12px;margin-top:4px;line-height:1.35;word-break:break-all}
.badge{display:inline-block;font-size:11px;padding:1px 6px;border-radius:10px;background:#ddd;margin-right:4px}
.badge.recut{background:#ffe08a}.badge.n{background:#eee}
.zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:50;cursor:zoom-out}
.zoom img{max-width:98vw;max-height:96vh}
.zoom.show{display:flex}
.viewer{position:fixed;inset:0;background:#111;display:none;flex-direction:column;z-index:60}
.viewer.show{display:flex}
.vbar{display:flex;gap:12px;align-items:center;padding:6px 12px;background:#222;color:#eee;font-size:13px}
.vbar #vhint{color:#aaa;margin-left:auto}
.vbar button.on{background:#2e7d32;color:#fff;border-color:#2e7d32}
.vstrip{display:flex;gap:8px;padding:6px 12px;background:#1a1a1a;overflow-x:auto;flex:none}
.vstrip .t{flex:none;width:150px;border:3px solid transparent;border-radius:6px;padding:2px;cursor:pointer;background:#2a2a2a;color:#ccc;font-size:11px}
.vstrip .t img{width:100%;display:block;border-radius:3px;background:#fff}
.vstrip .t .noimg{height:90px}
.vstrip .t.cur{border-color:#fff}
.vstrip .t.sel{background:#1b4d2b}
.vstrip .t .cap{padding:2px 2px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#osd{flex:1}
small.hint{color:var(--muted)}
#mode{font-size:12px;color:var(--muted)}
#err{color:var(--ex);font-size:13px}
</style></head><body>
<header>
  <b>Patho2Gliomas 代表 HE の選択</b>
  <span id="progress"></span>
  <span id="mode"></span>
  <span id="err"></span>
  <span id="filebtns" style="display:none">
    <button class="primary" onclick="exportCsv()">CSV を書き出す</button>
    <label class="btn">CSV を読み込む<input type="file" accept=".csv" style="display:none" onchange="importCsv(this.files[0])"></label>
  </span>
  <div class="filters">
    <select id="fGroup" onchange="render()"><option value="">全群</option></select>
    <select id="fState" onchange="render()"><option value="multi">HE が 2 枚以上だけ</option><option value="">全部</option><option value="todo">未選択だけ</option><option value="done">選択済みだけ</option></select>
  </div>
</header>
<main>
  <small class="hint">画像をクリックで WSI をそのまま拡大（← → で同じ症例の HE を切り替え）。代表は「代表にする」ボタン（カードでもビューアでも）。「除外」で症例ごと外す。HE が 1 枚の症例は最初から選択済み。</small>
  <div id="cases"></div>
</main>
<div class="zoom" id="zoom" onclick="this.classList.remove('show')"><img id="zoomimg"></div>
<div class="viewer" id="viewer">
  <div class="vbar"><button onclick="caseStep(-1)">◀ 前の症例</button><button onclick="caseStep(1)">次の症例 ▶</button><span id="vtitle"></span><button class="primary" id="vpick" onclick="viewerPick()">この切片を代表にする</button><button id="vexcl" onclick="viewerExclude()">症例を除外</button><span id="vhint">ホイールで拡大、ドラッグで移動。← → で同じ症例の HE、n / p で症例、Space で代表、Esc で閉じる</span><button onclick="closeViewer()">閉じる</button></div>
  <div class="vstrip" id="vstrip"></div>
  <div id="osd"></div>
</div>
<script>
const DATA = __DATA__;
const KEY = 'patho2_gliomas_review_v2';
let choice = {};      // case -> {file, note}; file 'EXCLUDE' = 症例除外
let server = false;   // serve で開いているか（クリックごとに CSV が書かれる）
function esc(s){ return String(s).replace(/[&<>"]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m])); }
function url(p){ return p.split('/').map(encodeURIComponent).join('/'); }   // ファイル名の # や空白を URL に通す
function saveLocal(){ try { localStorage.setItem(KEY, JSON.stringify(choice)); } catch (e) {} }
async function post(caseId, file){
  try {
    const r = await fetch('/choice', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({case: caseId, file})});
    const j = await r.json(); if (j.error) throw new Error(j.error);
    choice = j.choices; document.getElementById('err').textContent = '';
  } catch (e) { document.getElementById('err').textContent = '保存できませんでした: ' + e.message; }
  render();
}
function setChoice(caseId, file){   // file: ファイル名 / 'EXCLUDE' / null（解除）
  if (server) { post(caseId, file); return; }
  if (file === null) delete choice[caseId]; else choice[caseId] = {file, note: file === 'EXCLUDE' ? 'exclude' : 'manual'};
  saveLocal(); render();
}
function pick(caseId, file){ const cur = choice[caseId]; setChoice(caseId, cur && cur.file === file ? null : file); }
function exclude(caseId){ const cur = choice[caseId]; setChoice(caseId, cur && cur.file === 'EXCLUDE' ? null : 'EXCLUDE'); }
let visibleCases = [];   // いまのフィルタで表示している症例（ビューアの「次の症例」はこの順）
function render(){
  const g = document.getElementById('fGroup').value, st = document.getElementById('fState').value;
  const root = document.getElementById('cases'); root.innerHTML = ''; visibleCases = [];
  let done = 0, total = 0, excluded = 0, single = 0;
  for (const c of DATA.cases) {
    const ch = choice[c.case]; total++;
    if (ch && ch.file === 'EXCLUDE') excluded++; else if (ch) done++;
    if (ch && ch.note === 'single') single++;
    if (g && c.group !== g) continue;
    if (st === 'todo' && ch) continue;
    if (st === 'done' && !ch) continue;
    if (st === 'multi' && c.slides.length < 2) continue;
    visibleCases.push(c.case);
    const div = document.createElement('div');
    div.id = 'case-' + c.case;
    div.className = 'case' + (ch ? (ch.file === 'EXCLUDE' ? ' excluded' : ' done') : '');
    const state = ch ? (ch.file === 'EXCLUDE' ? '除外' : (ch.note === 'single' ? '1 枚なので自動' : '選択済み')) : '未選択';
    if (viewing) updateViewerPick();
    let h = `<h3><span>${esc(c.case)}</span><span class="meta">${esc(DATA.groups[c.group] || c.group)} ・ HE ${c.slides.length} 枚</span>` +
            `<span class="state">${state} <button onclick="exclude('${esc(c.case)}')">${ch && ch.file === 'EXCLUDE' ? '除外を戻す' : '除外'}</button></span></h3><div class="slides">`;
    for (const s of c.slides) {
      const sel = ch && ch.file === s.file;
      const badges = (s.recut ? '<span class="badge recut">再切片</span>' : '') +
                     (s.n_patches != null ? `<span class="badge n">${s.n_patches} patches</span>` : '<span class="badge n">未抽出</span>');
      const img = s.thumb ? `<img loading="lazy" src="${esc(url(s.thumb))}">` : `<div class="noimg">サムネイル未生成</div>`;
      const selbtn = `<button class="selbtn${sel ? ' on' : ''}" onclick="pick('${esc(c.case)}','${esc(s.file)}')">${sel ? '✔ 代表' : '代表にする'}</button>`;
      h += `<div class="slide${sel ? ' selected' : ''}"><div class="imgwrap" title="クリックで拡大" onclick="openViewer('${esc(c.case)}',${c.slides.indexOf(s)})">${img}</div><div class="cap">${selbtn}${badges}<br>${esc(s.file)}</div></div>`;
    }
    div.innerHTML = h + '</div>'; root.appendChild(div);
  }
  document.getElementById('progress').textContent = `選択済み ${done} / ${total} 症例（うち 1 枚で自動 ${single}、除外 ${excluded}）`;
}
function zoom(src){ document.getElementById('zoomimg').src = src; document.getElementById('zoom').classList.add('show'); }
// ---- WSI ビューア（serve のときだけ。Deep Zoom タイルをサーバーがその場で切る）----
let osd = null, viewing = null;   // viewing = {caseId, idx}（症例の HE の何枚目か）
function caseOf(caseId){ return DATA.cases.find(c => c.case === caseId); }
function loadOsd(){
  if (window.OpenSeadragon) return Promise.resolve();
  return new Promise((ok, ng) => { const s = document.createElement('script'); s.src = 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/openseadragon.min.js'; s.onload = ok; s.onerror = () => ng(new Error('OpenSeadragon を読み込めません（インターネットが要ります）')); document.head.appendChild(s); });
}
async function openViewer(caseId, idx){
  const c = caseOf(caseId); const s = c.slides[idx];
  if (!server) { if (s.thumb) zoom(url(s.thumb)); return; }
  try { await loadOsd(); } catch (e) { document.getElementById('err').textContent = e.message; if (s.thumb) zoom(url(s.thumb)); return; }
  viewing = {caseId, idx};
  document.getElementById('vtitle').textContent = `${caseId}（${DATA.groups[c.group] || c.group}） ${idx + 1} / ${c.slides.length} 枚目` + (s.recut ? ' ・ 再切片' : '');
  document.getElementById('viewer').classList.add('show');
  if (!osd) osd = OpenSeadragon({id: 'osd', prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/', showNavigator: true, navigatorPosition: 'BOTTOM_RIGHT', maxZoomPixelRatio: 2, animationTime: 0.3, imageLoaderLimit: 6});
  osd.open('/dzi/' + s.id + '.dzi');
  updateViewerPick();
}
function viewerStep(d){ if (!viewing) return; const c = caseOf(viewing.caseId); const n = (viewing.idx + d + c.slides.length) % c.slides.length; openViewer(viewing.caseId, n); }
function caseStep(d){
  if (!viewing) return;
  const list = visibleCases.length ? visibleCases : DATA.cases.map(c => c.case);
  let i = list.indexOf(viewing.caseId); if (i < 0) i = 0; else i = (i + d + list.length) % list.length;
  openViewer(list[i], 0);
  const el = document.getElementById('case-' + CSS.escape(list[i])); if (el) el.scrollIntoView({block: 'start'});
}
function renderStrip(){
  if (!viewing) return;
  const c = caseOf(viewing.caseId); const ch = choice[viewing.caseId];
  document.getElementById('vstrip').innerHTML = c.slides.map((s, i) => {
    const cls = 't' + (i === viewing.idx ? ' cur' : '') + (ch && ch.file === s.file ? ' sel' : '');
    const img = s.thumb ? `<img src="${esc(url(s.thumb))}">` : '<div class="noimg"></div>';
    return `<div class="${cls}" onclick="openViewer('${esc(c.case)}',${i})" title="${esc(s.file)}">${img}<div class="cap">${ch && ch.file === s.file ? '✔ ' : ''}${i + 1}. ${s.recut ? '再切片 ' : ''}${s.n_patches != null ? s.n_patches + 'p ' : ''}${esc(s.file)}</div></div>`;
  }).join('');
}
function viewerExclude(){ if (!viewing) return; exclude(viewing.caseId); setTimeout(updateViewerPick, 300); }
function updateViewerPick(){ if (!viewing) return; const c = caseOf(viewing.caseId); const file = c.slides[viewing.idx].file; const ch = choice[viewing.caseId]; const b = document.getElementById('vpick'); const on = ch && ch.file === file; b.textContent = on ? '✔ 代表に選択済み（クリックで解除）' : 'この切片を代表にする'; b.className = on ? 'on' : 'primary'; const ex = ch && ch.file === 'EXCLUDE'; document.getElementById('vexcl').textContent = ex ? '除外を戻す' : '症例を除外'; document.getElementById('vexcl').className = ex ? 'on' : ''; renderStrip(); }
function viewerPick(){ if (!viewing) return; const c = caseOf(viewing.caseId); pick(viewing.caseId, c.slides[viewing.idx].file); setTimeout(updateViewerPick, 300); }
function closeViewer(){ document.getElementById('viewer').classList.remove('show'); if (osd) osd.close(); viewing = null; }
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeViewer(); document.getElementById('zoom').classList.remove('show'); } else if (viewing && e.key === 'ArrowRight') viewerStep(1); else if (viewing && e.key === 'ArrowLeft') viewerStep(-1); else if (viewing && e.key === 'n') caseStep(1); else if (viewing && e.key === 'p') caseStep(-1); else if (viewing && (e.key === ' ' || e.key === 'Enter')) { e.preventDefault(); viewerPick(); } });
function exportCsv(){
  const rows = [['case','file','note']];
  for (const c of DATA.cases) { const ch = choice[c.case]; if (ch) rows.push([c.case, ch.file, ch.note]); }
  const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(',')).join('\n') + '\n';
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], {type:'text/csv'})); a.download = 'representatives.csv'; a.click();
}
function importCsv(f){
  if (!f) return; const rd = new FileReader();
  rd.onload = () => {
    const lines = rd.result.split(/\r?\n/).filter(l => l.trim()); const head = lines.shift().split(',').map(s => s.replace(/"/g,'').trim());
    const ci = head.indexOf('case'), fi = head.indexOf('file'), ni = head.indexOf('note');
    let n = 0;
    for (const l of lines) { const cells = l.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map(s => s.replace(/,$/,'').replace(/^"|"$/g,'').replace(/""/g,'"'));
      if (cells[ci]) { choice[cells[ci]] = {file: cells[fi], note: ni >= 0 && cells[ni] ? cells[ni] : 'manual'}; n++; } }
    saveLocal(); render(); alert(n + ' 症例を読み込みました');
  };
  rd.readAsText(f);
}
async function init(){
  const sel = document.getElementById('fGroup'); for (const [k, v] of Object.entries(DATA.groups)) { const o = document.createElement('option'); o.value = k; o.textContent = v; sel.appendChild(o); }
  try {
    if (location.protocol.startsWith('http')) { const r = await fetch('/state'); if (r.ok) { const j = await r.json(); choice = j.choices; server = true; document.getElementById('mode').textContent = 'クリックごとに ' + j.csv + ' に保存'; } }
  } catch (e) {}
  if (!server) {
    try { choice = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { choice = {}; }
    for (const c of DATA.cases) if (c.slides.length === 1 && !choice[c.case]) choice[c.case] = {file: c.slides[0].file, note: 'single'};
    document.getElementById('filebtns').style.display = ''; document.getElementById('mode').textContent = 'サーバーなし: 選択はこのブラウザに保存。終わったら「CSV を書き出す」→ ' + DATA.csv + ' に置く';
  }
  render();
}
init();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["thumbs", "serve", "html"])
    ap.add_argument("--workers", type=int, default=4, help="thumbs の並列数（NFS 読み）")
    ap.add_argument("--host", default=None, help="serve の bind 先（既定: config の patho2_gliomas.review_host、無ければ 127.0.0.1。SSH 先から開くなら 0.0.0.0）")
    ap.add_argument("--port", type=int, default=None, help="serve のポート（既定: config の patho2_gliomas.review_port）")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "thumbs":
        cmd_thumbs(cfg, args.workers)
    elif args.cmd == "serve":
        pcfg = cfg["patho2_gliomas"]
        cmd_serve(cfg, args.host or str(pcfg.get("review_host", "127.0.0.1")), args.port or int(pcfg.get("review_port", 8765)))
    else:
        cmd_html(cfg)


if __name__ == "__main__":
    main()
