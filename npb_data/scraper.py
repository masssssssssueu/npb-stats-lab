"""npb.jp (日本野球機構公式サイト) から成績データを取得するスクレイパー。

取得したページは data/raw/{year}/ 以下にHTMLキャッシュを保存し、
再実行時は再取得せずキャッシュを使う (npb.jpへの負荷軽減、オフライン再実行のため)。
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .teams import normalize_team

ROOT_SITE = "https://npb.jp"
BASE = "https://npb.jp/bis"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; npb-stats-research/1.0; +personal analysis project)"}

_REQUEST_INTERVAL_SEC = 1.0
_last_request_time = 0.0


def _polite_get(url: str) -> bytes:
    """npb.jpへの連続アクセス間隔を空けつつ取得する。"""
    global _last_request_time
    wait = _REQUEST_INTERVAL_SEC - (time.monotonic() - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    _last_request_time = time.monotonic()
    resp.raise_for_status()
    return resp.content


def fetch_html(url: str, *, use_cache: bool = True) -> str:
    """URLのHTMLをUTF-8文字列で取得する。ローカルキャッシュがあればそれを使う。"""
    cache_path = RAW_CACHE_DIR / re.sub(r"^https?://", "", url).replace("/", "_")
    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    content = _polite_get(url)
    text = content.decode("utf-8")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def _read_tables(html: str) -> list[pd.DataFrame]:
    return pd.read_html(io.StringIO(html))


_NAME_TEAM_RE = re.compile(r"^(?P<name>.+?)\((?P<team>.+?)\)\s*$")


def _split_name_team(raw: str) -> tuple[str, str]:
    m = _NAME_TEAM_RE.match(str(raw).strip())
    if not m:
        return str(raw).strip(), ""
    name = m.group("name").replace("　", " ").strip()
    team = normalize_team(m.group("team"))
    return name, team


def fetch_batting(year: int, league: str, *, use_cache: bool = True) -> pd.DataFrame:
    """個人打撃成績 (規定打席以上)。league: 'c' (セ) or 'p' (パ)"""
    url = f"{BASE}/{year}/stats/bat_{league}.html"
    html = fetch_html(url, use_cache=use_cache)
    df = _read_tables(html)[0]
    names_teams = df["選手"].map(_split_name_team)
    df["選手"] = names_teams.map(lambda t: t[0])
    df["球団"] = names_teams.map(lambda t: t[1])
    df["年度"] = year
    df["リーグ"] = "セ" if league == "c" else "パ"
    return df


def fetch_pitching(year: int, league: str, *, use_cache: bool = True) -> pd.DataFrame:
    """個人投手成績 (規定投球回以上)。league: 'c' (セ) or 'p' (パ)"""
    url = f"{BASE}/{year}/stats/pit_{league}.html"
    html = fetch_html(url, use_cache=use_cache)
    df = _read_tables(html)[0]
    names_teams = df["投手"].map(_split_name_team)
    df["投手"] = names_teams.map(lambda t: t[0])
    df["球団"] = names_teams.map(lambda t: t[1])
    df["年度"] = year
    df["リーグ"] = "セ" if league == "c" else "パ"
    return df


def fetch_team_batting(year: int, league: str, *, use_cache: bool = True) -> pd.DataFrame:
    url = f"{BASE}/{year}/stats/tmb_{league}.html"
    html = fetch_html(url, use_cache=use_cache)
    df = _read_tables(html)[0]
    df["チーム"] = df["チーム"].map(normalize_team)
    df["年度"] = year
    return df


def fetch_team_pitching(year: int, league: str, *, use_cache: bool = True) -> pd.DataFrame:
    url = f"{BASE}/{year}/stats/tmp_{league}.html"
    html = fetch_html(url, use_cache=use_cache)
    df = _read_tables(html)[0]
    df["チーム"] = df["チーム"].map(normalize_team)
    df["年度"] = year
    return df


def fetch_standings(year: int, league: str, *, use_cache: bool = True) -> pd.DataFrame:
    """順位表。1つ目のテーブルがリーグ順位、2つ目は交流戦内訳なので使わない。"""
    url = f"{BASE}/{year}/stats/std_{league}.html"
    html = fetch_html(url, use_cache=use_cache)
    df = _read_tables(html)[0]
    df["チーム"] = df["チーム"].map(normalize_team)
    df["年度"] = year
    df["リーグ"] = "セ" if league == "c" else "パ"
    return df


def fetch_all(year: int, *, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """1年度分の全データを取得して辞書で返す。"""
    batting = pd.concat(
        [fetch_batting(year, lg, use_cache=use_cache) for lg in ("c", "p")], ignore_index=True
    )
    pitching = pd.concat(
        [fetch_pitching(year, lg, use_cache=use_cache) for lg in ("c", "p")], ignore_index=True
    )
    team_batting = pd.concat(
        [fetch_team_batting(year, lg, use_cache=use_cache) for lg in ("c", "p")], ignore_index=True
    )
    team_pitching = pd.concat(
        [fetch_team_pitching(year, lg, use_cache=use_cache) for lg in ("c", "p")], ignore_index=True
    )
    standings = pd.concat(
        [fetch_standings(year, lg, use_cache=use_cache) for lg in ("c", "p")], ignore_index=True
    )
    return {
        "batting": batting,
        "pitching": pitching,
        "team_batting": team_batting,
        "team_pitching": team_pitching,
        "standings": standings,
    }


TEAM_CODE = {
    "阪神": "t", "巨人": "g", "DeNA": "db", "広島": "c", "中日": "d", "ヤクルト": "s",
    "ソフトバンク": "h", "西武": "l", "日本ハム": "f", "オリックス": "b", "ロッテ": "m", "楽天": "e",
}


def fetch_roster(team: str, *, use_cache: bool = True) -> pd.DataFrame:
    """球団の選手名鑑 (支配下選手・育成選手) を取得する。個人ページへのリンクIDを含む。"""
    code = TEAM_CODE[team]
    html = fetch_html(f"{ROOT_SITE}/bis/teams/rst_{code}.html", use_cache=use_cache)
    soup = BeautifulSoup(html, "lxml")

    rows = []
    for sub in soup.select(".rosterSub"):
        category = sub.get_text(strip=True)
        table = sub.find_next("table", class_="rosterlisttbl")
        if not table:
            continue
        position = None
        for tr in table.select("tr"):
            pos_th = tr.select_one("th.rosterPos")
            if pos_th:
                position = pos_th.get_text(strip=True)
                continue
            a = tr.select_one('a[href^="/bis/players/"]')
            if not a:
                continue
            player_id = a["href"].rsplit("/", 1)[-1].replace(".html", "")
            name = a.get_text(strip=True).replace("　", " ")
            number_td = tr.select_one("td")
            number = number_td.get_text(strip=True) if number_td else ""
            rows.append({
                "team": team, "category": category, "position": position,
                "player_id": player_id, "name": name, "number": number,
            })
    return pd.DataFrame(rows)


def player_photo_url(team: str, number: str, player_id: str, year: int) -> str | None:
    """選手の顔写真URLを組み立てる (存在確認済みのURLパターン: 予告先発ページで実際に使われている)。"""
    code = TEAM_CODE.get(team)
    if not code or not number or not player_id:
        return None
    try:
        num = int(number)
    except ValueError:
        return None
    return f"{ROOT_SITE.replace('https://npb.jp', 'https://p.npb.jp')}/players_photo/{year}/180/{code}/{num:03d}_{player_id}.jpg"


def team_logo_url(team: str, year: int, size: str = "m") -> str | None:
    """球団ロゴ画像URLを組み立てる。マスコットキャラクターの画像はnpb.jp上に無いため、代わりに球団ロゴを使う。"""
    code = TEAM_CODE.get(team)
    if not code:
        return None
    return f"https://p.npb.jp/img/common/logo/{year}/logo_{code}_{size}.gif"


_PLAYER_PAGE_PITCHING_RENAME = {"H": "ホールド", "HP": "ＨＰ"}


def _parse_year_row(table, year: int, header_rename: dict[str, str] | None = None) -> dict | None:
    if table is None:
        return None
    header_rename = header_rename or {}
    headers = [header_rename.get(th.get_text(strip=True), th.get_text(strip=True)) for th in table.select("thead th")]
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td", recursive=False)
        if not cells or cells[0].get_text(strip=True) != str(year):
            continue
        record = {}
        for header, td in zip(headers, cells):
            inning_table = td.select_one("table.table_inning")
            if inning_table:
                whole = inning_table.select_one("th")
                frac = inning_table.select_one("td")
                whole_n = re.sub(r"\D", "", whole.get_text(strip=True)) if whole else ""
                frac_n = re.sub(r"\D", "", frac.get_text(strip=True)) if frac else ""
                record[header] = f"{whole_n or '0'}.{frac_n or '0'}"
            else:
                record[header] = td.get_text(strip=True)
        return record
    return None


def fetch_player_season(player_id: str, year: int, *, use_cache: bool = True) -> dict:
    """個人年度別成績ページから指定年度の打撃・投手成績を取得する(規定打席・投球回未満の選手も含む)。

    このページには「故意四」列が無いため、後段の互換性のために0で補う。
    """
    html = fetch_html(f"{ROOT_SITE}/bis/players/{player_id}.html", use_cache=use_cache)
    soup = BeautifulSoup(html, "lxml")
    pitching = _parse_year_row(soup.select_one("table#tablefix_p"), year, _PLAYER_PAGE_PITCHING_RENAME)
    batting = _parse_year_row(soup.select_one("table#tablefix_b"), year)
    if pitching is not None:
        pitching.setdefault("故意四", "0")
    if batting is not None:
        batting.setdefault("故意四", "0")
    return {"batting": batting, "pitching": pitching}


_FINISHED_KEYWORDS = ("試合終了", "中止", "ノーゲーム", "コールド")


def fetch_today_status() -> dict:
    """トップページの試合速報ウィジェットから、本日の各試合の終了状況を取得する。

    日次自動更新のトリガー判定に使う。キャッシュはしない(常に最新を見る必要があるため)。
    """
    html = fetch_html(f"{ROOT_SITE}/", use_cache=False)
    soup = BeautifulSoup(html, "lxml")

    date_box = soup.select_one(".score_wrap .score_box.date")
    date_label = date_box.get_text(" ", strip=True) if date_box else ""

    games = []
    for box in soup.select(".score_wrap > .score_box"):
        classes = box.get("class", [])
        if "date" in classes or "detail" in classes:
            continue
        imgs = box.select("img")
        if len(imgs) < 2:
            continue
        team1 = normalize_team(imgs[0].get("alt", ""))
        team2 = normalize_team(imgs[1].get("alt", ""))
        state_tag = box.select_one(".state")
        state = re.sub(r"\s+", " ", state_tag.get_text(" ", strip=True)) if state_tag else ""
        finished = any(k in state for k in _FINISHED_KEYWORDS)
        games.append({"team1": team1, "team2": team2, "state": state, "finished": finished})

    return {"date_label": date_label, "games": games}


def fetch_probable_starters() -> dict:
    """次回開催日の予告先発情報 (/announcement/starter/) を取得する。日次で変わるためキャッシュしない。"""
    html = fetch_html(f"{ROOT_SITE}/announcement/starter/", use_cache=False)
    soup = BeautifulSoup(html, "lxml")

    date_label = ""
    h4 = soup.select_one(".contents h4")
    if h4:
        date_label = h4.get_text(strip=True)

    matchups = []
    for unit in soup.select("section.starting_wrap_cl div.unit, section.starting_wrap_pl div.unit"):
        left_img = unit.select_one(".team_left img")
        right_img = unit.select_one(".team_right img")
        if not left_img or not right_img:
            continue
        home_team = normalize_team(left_img.get("alt", ""))
        away_team = normalize_team(right_img.get("alt", ""))
        home_span = unit.select_one(".team_left span")
        away_span = unit.select_one(".team_right span")
        home_pitcher = home_span.get_text(strip=True).replace("　", " ") if home_span else None
        away_pitcher = away_span.get_text(strip=True).replace("　", " ") if away_span else None
        info = unit.select_one(".info")
        info_text = re.sub(r"\s+", " ", info.get_text(" ", strip=True)) if info else ""
        matchups.append({
            "home_team": home_team, "away_team": away_team,
            "home_pitcher": home_pitcher, "away_pitcher": away_pitcher,
            "info": info_text,
        })

    return {"date_label": date_label, "matchups": matchups}


def fetch_schedule_month(year: int, month: int) -> pd.DataFrame:
    """指定月の試合日程・結果 (試合速報を含む) を取得する。シーズン進行中は変わるためキャッシュしない。"""
    url = f"{ROOT_SITE}/games/{year}/schedule_{month:02d}_detail.html"
    html = fetch_html(url, use_cache=False)
    soup = BeautifulSoup(html, "lxml")

    rows = []
    for tr in soup.select("#schedule_detail table tbody tr"):
        team1_tag = tr.select_one(".team1")
        team2_tag = tr.select_one(".team2")
        if not team1_tag or not team2_tag:
            continue
        score1_tag = tr.select_one(".score1")
        score2_tag = tr.select_one(".score2")
        s1 = score1_tag.get_text(strip=True) if score1_tag else ""
        s2 = score2_tag.get_text(strip=True) if score2_tag else ""
        completed = s1.isdigit() and s2.isdigit()
        link_tag = tr.select_one('a[href^="/scores/"]')
        game_url = f"{ROOT_SITE}{link_tag['href']}" if link_tag else None
        rows.append({
            "team1": normalize_team(team1_tag.get_text(strip=True)),
            "team2": normalize_team(team2_tag.get_text(strip=True)),
            "score1": int(s1) if completed else None,
            "score2": int(s2) if completed else None,
            "completed": completed,
            "game_url": game_url,
        })
    return pd.DataFrame(rows)


def fetch_box_score(game_url: str, *, use_cache: bool = True) -> dict:
    """試合のボックススコア(個人成績)を取得する。終了済みの試合は結果が変わらないためキャッシュする。"""
    url = game_url.rstrip("/") + "/box.html"
    html = fetch_html(url, use_cache=use_cache)
    soup = BeautifulSoup(html, "lxml")

    def parse_batting(table_id: str) -> list[dict]:
        table = soup.select_one(f"table#{table_id}")
        if not table:
            return []
        out = []
        for tr in table.select("tbody tr"):
            a = tr.select_one("td.player a")
            if not a:
                continue
            cells = tr.find_all("td", recursive=False)
            if len(cells) < 7:
                continue
            player_id = a["href"].rsplit("/", 1)[-1].replace(".html", "")
            try:
                ab, r, h, rbi, sb = (int(c.get_text(strip=True) or 0) for c in cells[3:8])
            except ValueError:
                continue
            out.append({"player_id": player_id, "name": a.get_text(strip=True), "AB": ab, "R": r, "H": h, "RBI": rbi, "SB": sb})
        return out

    def parse_pitching(table_id: str) -> list[dict]:
        table = soup.select_one(f"table#{table_id}")
        if not table:
            return []
        out = []
        for tr in table.select("tbody tr"):
            a = tr.select_one("td.player a")
            if not a:
                continue
            player_id = a["href"].rsplit("/", 1)[-1].replace(".html", "")
            ip_table = tr.select_one("table.table_inning")
            if ip_table:
                whole = ip_table.select_one("th")
                frac = ip_table.select_one("td")
                whole_n = re.sub(r"\D", "", whole.get_text(strip=True)) if whole else ""
                frac_n = re.sub(r"\D", "", frac.get_text(strip=True)) if frac else ""
                ip = f"{whole_n or '0'}.{frac_n or '0'}"
            else:
                ip = "0.0"
            # td順(0始まり): [0]勝敗記号 [1]投手名 [2]投球数 [3]打者 [4]投球回 [5]安打 [6]本塁打
            #               [7]四球 [8]死球 [9]三振 [10]暴投 [11]ボーク [12]失点 [13]自責点
            cells = tr.find_all("td", recursive=False)
            if len(cells) < 14:
                continue

            def _n(idx: int) -> int:
                text = re.sub(r"\D", "", cells[idx].get_text(strip=True))
                return int(text) if text else 0

            out.append({
                "player_id": player_id, "name": a.get_text(strip=True), "IP": ip,
                "H": _n(5), "HR": _n(6), "BB": _n(7), "HBP": _n(8), "SO": _n(9),
                "R": _n(12), "ER": _n(13),
            })
        return out

    return {
        "team1_batting": parse_batting("tablefix_t_b"),
        "team1_pitching": parse_pitching("tablefix_t_p"),
        "team2_batting": parse_batting("tablefix_b_b"),
        "team2_pitching": parse_pitching("tablefix_b_p"),
    }
