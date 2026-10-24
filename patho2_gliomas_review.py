"""patho2_gliomas_review.py — 院内グリオーマの代表 HE を目で選ぶための簡易ツール（サムネイル + 静的 HTML）。

1 症例 1 枚の代表は patho2_gliomas_select.py の規則（再切片 → 組織パッチ数）で決まるが、腫瘍が写っていない切片を避けるには
目で見て選ぶのが確実なので、症例ごとに HE のサムネイルを並べ、クリックで代表を選ぶ HTML を作る。
選択は CSV（case, file, note）に書き出し、config の patho2_gliomas.override に置けば select.py がそれを優先する
（file が EXCLUDE の症例は除外）。

手順:
    thumbs   slides.csv の HE 全部について WSI からサムネイル（長辺 thumb_px、JPEG）を作る → review/thumbs/<群>/<症例>/<元名>.jpg。既存はスキップ
    html     review/index.html を組む（slides.csv と、あれば representatives.csv の規則の選択を既定として埋め込む）
    all      thumbs → html

使い方（ken 側）:
    out/patho2_gliomas/review/index.html をブラウザで開く（Sync 経由でローカルでも、画像は相対パス）。
    サムネイルをクリックで代表、「除外」で症例ごと外す。選択はブラウザに自動保存される。
    「CSV を書き出す」で representative_override.csv を保存 → data/patho2_gliomas/representative_override.csv に置く →
    uv run python patho2_gliomas_select.py。「CSV を読み込む」で前回の CSV から続きができる。

Usage:
    uv run python -u patho2_gliomas_review.py thumbs
    uv run python -u patho2_gliomas_review.py html
    uv run python -u patho2_gliomas_review.py all
"""
from __future__ import annotations

import argparse
import html
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from utils.loader import load_config

GROUP_NAMES = {"A": "A: IDH 変異型星細胞腫", "G": "G: IDH 野生型", "O": "O: 乏突起膠腫", "L": "L: リンパ腫", "M": "M: 転移"}


def review_dir(cfg: dict) -> Path:
    return Path(cfg["output_dir"]) / "patho2_gliomas" / "review"


def thumb_path(rdir: Path, r) -> Path:
    return rdir / "thumbs" / r.group / str(r.case) / (Path(r.file).stem + ".jpg")


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
    pcfg = cfg["patho2_gliomas"]
    rdir = review_dir(cfg)
    slides = pd.read_csv(Path(cfg["output_dir"]) / "patho2_gliomas" / "slides.csv")
    he = slides[slides["is_he"]]
    long_px = int(pcfg.get("thumb_px", 1400))
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


def cmd_html(cfg: dict) -> None:
    out_dir = Path(cfg["output_dir"]) / "patho2_gliomas"
    rdir = review_dir(cfg)
    slides = pd.read_csv(out_dir / "slides.csv")
    he = slides[slides["is_he"]].copy()
    if "n_patches" not in he:          # --dry-run の index には抽出の列が無い
        he["n_patches"] = float("nan")
    he["n_patches"] = pd.to_numeric(he["n_patches"], errors="coerce")
    defaults: dict[str, dict] = {}
    rep_path = out_dir / "representatives.csv"
    if rep_path.exists():
        reps = pd.read_csv(rep_path)
        for r in reps[reps["status"] == "selected"].itertuples(index=False):
            defaults[str(r.case)] = {"file": r.file, "reason": r.reason}

    cases = []
    for (group, case), df in he.groupby(["group", "case"], sort=False):
        items = []
        for r in df.sort_values("file").itertuples(index=False):
            tp = thumb_path(rdir, r)
            items.append({
                "file": r.file, "stain": "" if pd.isna(r.stain) else str(r.stain), "recut": bool(r.recut),
                "n_patches": None if pd.isna(r.n_patches) else int(r.n_patches),
                "thumb": str(tp.relative_to(rdir)) if tp.exists() else None,
            })
        d = defaults.get(str(case))
        cases.append({"group": group, "label": df["label"].iloc[0], "case": str(case), "slides": items,
                      "default": d["file"] if d else None, "default_reason": d["reason"] if d else None})
    order = list(cfg["patho2_gliomas"].get("priority", ["A", "G", "O", "L", "M"]))
    cases.sort(key=lambda c: (order.index(c["group"]) if c["group"] in order else 99, c["case"]))

    data_json = json.dumps({"cases": cases, "groups": {g: GROUP_NAMES.get(g, g) for g in order},
                            "generated": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False)
    page = HTML_TEMPLATE.replace("__DATA__", data_json.replace("</", "<\\/"))
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "index.html").write_text(page, encoding="utf-8")
    n_thumb = sum(1 for c in cases for s in c["slides"] if s["thumb"])
    print(f"-> {rdir / 'index.html'}: {len(cases)} cases, {sum(len(c['slides']) for c in cases)} HE "
          f"(thumbnails {n_thumb}, rule defaults {sum(1 for c in cases if c['default'])})")


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
.slide{width:300px;border:3px solid transparent;border-radius:8px;padding:4px;cursor:pointer;background:#f3f3f3}
.slide:hover{background:#e9eef5}
.slide.selected{border-color:var(--sel);background:#e3eef9}
.slide img{width:100%;height:auto;display:block;border-radius:4px;background:#fff}
.slide .noimg{height:200px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px}
.slide .cap{font-size:12px;margin-top:4px;line-height:1.35;word-break:break-all}
.badge{display:inline-block;font-size:11px;padding:1px 6px;border-radius:10px;background:#ddd;margin-right:4px}
.badge.recut{background:#ffe08a}.badge.rule{background:#cfe8ff}.badge.n{background:#eee}
.zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:50;cursor:zoom-out}
.zoom img{max-width:98vw;max-height:96vh}
.zoom.show{display:flex}
small.hint{color:var(--muted)}
</style></head><body>
<header>
  <b>Patho2Gliomas 代表 HE の選択</b>
  <span id="progress"></span>
  <button class="primary" onclick="exportCsv()">CSV を書き出す</button>
  <label class="btn">CSV を読み込む<input type="file" accept=".csv" style="display:none" onchange="importCsv(this.files[0])"></label>
  <button onclick="clearAll()">選択を全部消す</button>
  <div class="filters">
    <select id="fGroup" onchange="render()"><option value="">全群</option></select>
    <select id="fState" onchange="render()"><option value="">全部</option><option value="todo">未選択だけ</option><option value="multi">HE が 2 枚以上だけ</option><option value="done">選択済みだけ</option></select>
  </div>
</header>
<main>
  <small class="hint">クリックで代表に。もう一度クリックで解除。「規則」は patho2_gliomas_select.py の既定（再切片 → 組織パッチ数最大）。画像をダブルクリックで拡大。選択はこのブラウザに自動保存。書き出した CSV は data/patho2_gliomas/representative_override.csv に置く。</small>
  <div id="cases"></div>
</main>
<div class="zoom" id="zoom" onclick="this.classList.remove('show')"><img id="zoomimg"></div>
<script>
const DATA = __DATA__;
const KEY = 'patho2_gliomas_review_v1';
let choice = {};  // case -> {file, note} ; file 'EXCLUDE' = 症例除外
try { choice = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { choice = {}; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(choice)); } catch (e) {} }
function effective(c){ if (choice[c.case]) return choice[c.case]; if (c.default) return {file:c.default, note:'rule:'+c.default_reason}; return null; }
function pick(caseId, file){ const cur = choice[caseId]; if (cur && cur.file === file) delete choice[caseId]; else choice[caseId] = {file, note:'manual'}; save(); render(); }
function exclude(caseId){ const cur = choice[caseId]; if (cur && cur.file === 'EXCLUDE') delete choice[caseId]; else choice[caseId] = {file:'EXCLUDE', note:'manual'}; save(); render(); }
function esc(s){ return String(s).replace(/[&<>"]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m])); }
function render(){
  const g = document.getElementById('fGroup').value, st = document.getElementById('fState').value;
  const root = document.getElementById('cases'); root.innerHTML = '';
  let done = 0, total = 0, manual = 0, excluded = 0;
  for (const c of DATA.cases) {
    const eff = effective(c); total++;
    if (eff && eff.file === 'EXCLUDE') excluded++; else if (eff) done++;
    if (choice[c.case]) manual++;
    if (g && c.group !== g) continue;
    const isDone = !!eff;
    if (st === 'todo' && isDone) continue;
    if (st === 'done' && !isDone) continue;
    if (st === 'multi' && c.slides.length < 2) continue;
    const div = document.createElement('div');
    div.className = 'case' + (eff ? (eff.file === 'EXCLUDE' ? ' excluded' : ' done') : '');
    const state = eff ? (eff.file === 'EXCLUDE' ? '除外' : (eff.note === 'manual' ? '手動で選択' : '規則の既定')) : '未選択';
    let h = `<h3><span>${esc(c.case)}</span><span class="meta">${esc(DATA.groups[c.group] || c.group)} ・ HE ${c.slides.length} 枚</span>` +
            `<span class="state">${state} <button onclick="exclude('${esc(c.case)}')">${eff && eff.file === 'EXCLUDE' ? '除外を戻す' : '除外'}</button></span></h3><div class="slides">`;
    for (const s of c.slides) {
      const sel = eff && eff.file === s.file;
      const badges = (s.recut ? '<span class="badge recut">再切片</span>' : '') + (c.default === s.file ? '<span class="badge rule">規則</span>' : '') +
                     (s.n_patches != null ? `<span class="badge n">${s.n_patches} patches</span>` : '<span class="badge n">未抽出</span>');
      const img = s.thumb ? `<img loading="lazy" src="${esc(s.thumb)}" ondblclick="event.stopPropagation();zoom('${esc(s.thumb)}')">` : '<div class="noimg">サムネイル未生成</div>';
      h += `<div class="slide${sel ? ' selected' : ''}" onclick="if(event.detail>1)return;pick('${esc(c.case)}','${esc(s.file)}')">${img}<div class="cap">${badges}<br>${esc(s.file)}</div></div>`;
    }
    div.innerHTML = h + '</div>'; root.appendChild(div);
  }
  document.getElementById('progress').textContent = `選択済み ${done} / ${total} 症例（手動 ${manual}、除外 ${excluded}）・生成 ${DATA.generated}`;
}
function zoom(src){ document.getElementById('zoomimg').src = src; document.getElementById('zoom').classList.add('show'); }
function exportCsv(){
  const rows = [['case','file','note']];
  for (const c of DATA.cases) { const eff = effective(c); if (eff) rows.push([c.case, eff.file, eff.note]); }
  const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(',')).join('\n') + '\n';
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], {type:'text/csv'})); a.download = 'representative_override.csv'; a.click();
}
function importCsv(f){
  if (!f) return; const rd = new FileReader();
  rd.onload = () => {
    const lines = rd.result.split(/\r?\n/).filter(l => l.trim()); const head = lines.shift().split(',').map(s => s.replace(/"/g,'').trim());
    const ci = head.indexOf('case'), fi = head.indexOf('file'), ni = head.indexOf('note');
    let n = 0;
    for (const l of lines) { const cells = l.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map(s => s.replace(/,$/,'').replace(/^"|"$/g,'').replace(/""/g,'"'));
      if (cells[ci]) { choice[cells[ci]] = {file: cells[fi], note: ni >= 0 && cells[ni] ? cells[ni] : 'manual'}; n++; } }
    save(); render(); alert(n + ' 症例を読み込みました');
  };
  rd.readAsText(f);
}
function clearAll(){ if (confirm('このブラウザに保存した選択を全部消します。よいですか？')) { choice = {}; save(); render(); } }
const sel = document.getElementById('fGroup'); for (const [k, v] of Object.entries(DATA.groups)) { const o = document.createElement('option'); o.value = k; o.textContent = v; sel.appendChild(o); }
render();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["thumbs", "html", "all"])
    ap.add_argument("--workers", type=int, default=4, help="サムネイル生成の並列数（NFS 読み）")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd in ("thumbs", "all"):
        cmd_thumbs(cfg, args.workers)
    if args.cmd in ("html", "all"):
        cmd_html(cfg)


if __name__ == "__main__":
    main()
