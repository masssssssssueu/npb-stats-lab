"""その日のNPB全試合が終了していたら、サイトを1回だけ再生成する。

Windowsタスクスケジューラから夜間に繰り返し呼び出される想定(例: 21:00〜23:30の間30分おき)。
- まだ試合が残っていれば何もせず終了する(次の実行タイミングで再チェックされる)。
- 全試合終了を確認したら build_site.py を実行し、本日分は完了マーカーを書いて重複実行を防ぐ。
- 試合が1つも無い日(オフ日)は即座に更新してよいと判断する。
"""
from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

from npb_data import scraper

ROOT = Path(__file__).resolve().parent
MARKER = ROOT / "data" / "last_build_date.txt"
LOG = ROOT / "data" / "daily_update.log"


def _log(message: str) -> None:
    line = f"{date.today().isoformat()} {message}"
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def already_built_today() -> bool:
    if not MARKER.exists():
        return False
    return MARKER.read_text(encoding="utf-8").strip() == date.today().isoformat()


def all_games_finished() -> bool:
    status = scraper.fetch_today_status()
    if not status["games"]:
        return True
    return all(g["finished"] for g in status["games"])


def main() -> None:
    if already_built_today():
        _log("本日は既に更新済みです。スキップします。")
        return

    if not all_games_finished():
        _log("本日の試合がまだ終了していません。今回はスキップします。")
        return

    _log("本日の全試合が終了しました。サイトを更新します。")
    result = subprocess.run(
        [sys.executable, str(ROOT / "build_site.py"), str(date.today().year)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        _log(f"更新に失敗しました (exit code {result.returncode})。マーカーは書き込みません。")
        return

    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(date.today().isoformat(), encoding="utf-8")
    _log("更新が完了しました。")


if __name__ == "__main__":
    main()
