"""figs.all — 登録済みの図表をすべて作る。

先に fig/・tables/ の中の「figs が出したファイル」（番号付きの幹 `fig\\d+_<name>`・`table\\d+_<name>`・
`figS\\d+_<name>`・`tableS\\d+_<name>` と、旧命名の出力）だけを消してから作り直す。番号が変わっても古い番号のファイルが残らない。
それ以外のファイルは消さない。スクリプトが無い登録名（手描きの design）は飛ばす。

Usage:
    uv run python -m figs.all
"""
from __future__ import annotations

import importlib
import importlib.util
import re

from figs import MAIN, SUPP, SUPP_TABLES, TABLES, label
from figs.common import config, fig_dir, table_dir

# 番号ベースだった頃の出力名（掃除の対象）
LEGACY = re.compile(r"^((fig[2-5]|fig5_pairs|fig5b_sft_distance|figS_dendrogram_original|table[12])\.(png|pdf|csv|md))$")  # _preview_* は見比べ用なので消さない


def _clean(cfg: dict) -> None:
    names = "|".join(re.escape(n) for n in MAIN + TABLES + SUPP + SUPP_TABLES)
    ours = re.compile(rf"^(fig|table|figS|tableS)\d+_({names})(\.|_)")
    for d in (fig_dir(cfg), table_dir(cfg)):
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and (ours.match(p.name) or LEGACY.match(p.name)):
                p.unlink()
                print(f"  removed: {p}")


def main() -> None:
    cfg = config()
    print("[clean]")
    _clean(cfg)
    for name in MAIN + TABLES + SUPP + SUPP_TABLES:
        if importlib.util.find_spec(f"figs.{name}") is None:
            print(f"[{label(name)}] {name}: スクリプトなし（手描き / 未作成）— skip")
            continue
        importlib.import_module(f"figs.{name}").main()


if __name__ == "__main__":
    main()
