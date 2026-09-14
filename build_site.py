"""NPB成績データを取得し、セイバーメトリクス分析・順位予想を行って静的サイトを生成する。

使い方:
    .venv\\Scripts\\python.exe build_site.py [年度]

出力先: site/ (ブラウザで index.html を直接開けば閲覧可能)
"""
from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

from npb_data import predictions, sabermetrics, scraper
from npb_data.teams import ALL_TEAMS, CENTRAL_TEAMS, PACIFIC_TEAMS, TEAM_COLORS

ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "site"
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2026

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}


def style_fig(fig: go.Figure, height: int = 440) -> str:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a1d23", family="Segoe UI, Hiragino Sans, sans-serif"),
        margin=dict(l=10, r=10, t=44, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#dfe3e8", zerolinecolor="#dfe3e8")
    fig.update_yaxes(gridcolor="#dfe3e8", zerolinecolor="#dfe3e8")
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def df_to_html(df: pd.DataFrame, classes: str = "stats") -> str:
    df = glossarize_columns(df)
    return df.to_html(index=False, classes=classes, border=0, escape=False, na_rep="-")


GLOSSARY = {
    "OPS": "出塁率+長打率。打者の総合的な得点貢献度を表す代表的な指標。",
    "ISO": "長打率-打率。単打を除いた純粋な長打力を表す。",
    "wOBA": "四球・単打・二塁打・三塁打・本塁打に加重係数をかけて算出する打撃総合指標(簡易版・MLB由来の係数を流用)。",
    "RC": "Runs Created。その打者が生み出したと推定される得点数。",
    "RC27": "27アウト(1試合分)を消費する間に、その打者のペースで生み出す推定得点数。",
    "BABIP": "Batting Average on Balls In Play。フィールド内に飛んだ打球がヒットになった割合。",
    "K%": "三振率。打席(打者)のうち三振に終わった割合。",
    "BB%": "四球率。打席(打者)のうち四球で出塁した割合。",
    "WHIP": "1投球回あたりに許した走者数 (与四球+被安打)÷投球回。",
    "FIP": "Fielding Independent Pitching。守備の巧拙を除き、本塁打・四死球・奪三振だけで防御率相当を算出する指標。",
    "K/9": "9イニングあたりの奪三振数。",
    "BB/9": "9イニングあたりの与四球数。",
    "K/BB": "奪三振数を与四球数で割った値。制球と奪三振能力のバランスを示す。",
    "ピタゴラス勝率": "得点と失点から算出する期待勝率。実際の勝率とのズレは「運」の目安になる。",
    "運(実勝利-期待勝利)": "実際の勝利数とピタゴラス勝率が示す期待勝利数の差。プラスは接戦を拾って勝てている、マイナスはその逆。",
    "期待勝利数": "ピタゴラス勝率に試合数を掛けた、得失点から見た期待される勝利数。",
    "モンテカルロ・シミュレーション": "乱数を使って残り試合の結果を何度も仮想的に試行し、最終順位の確率分布を推定する手法。",
    "log5法": "Bill James考案の、2チームの勝率から対戦時の勝率を推定する計算式。",
    "安定化点": "そのスタッツが選手の実力をある程度反映し始める目安のサンプルサイズ(打席数・投球回)。値が小さいほどリーグ平均へ強く回帰させる。",
    "FIP定数": "そのリーグ・年度の実際の防御率に合わせてFIPの基準点を調整するための定数(cFIP)。",
    "WAR": "Wins Above Replacement(控え選手と比較した勝利貢献度)。baseballhub.appの簡易算出法(OPSから概算したwRC+・出場試合数・盗塁数・守備位置補正のみで計算)を採用した近似値。UZR等の詳細な守備データは使っていないため、真の意味でのWARとは異なる参考値。投手版は算出方法が確立されていないため未対応。",
    "wRC+": "打撃力を100を基準にした指数にした参考値。baseballhub.app方式(100+(OPS-0.700)×220、30〜220にクリップ)による簡易版で、リーグ・球場補正は行っていない。",
}


def term(key: str, display: str | None = None) -> str:
    """用語をホバー(タップ)すると解説が出るスパンでラップする。GLOSSARY未登録ならそのまま返す。"""
    tip = GLOSSARY.get(key)
    display = display if display is not None else key
    if not tip:
        return display
    return f'<span class="term" tabindex="0">{display}<span class="tip">{tip}</span></span>'


def glossarize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名がGLOSSARYに載っている場合、見出しにホバー解説を付ける。"""
    return df.rename(columns={c: term(c) for c in df.columns if c in GLOSSARY})


def slugify(team: str, name: str) -> str:
    s = re.sub(r"[\s　]+", "", f"{team}-{name}")
    return re.sub(r"[\\/:*?\"<>|]", "", s)


def linkify_players(df: pd.DataFrame, name_col: str, prefix: str = "players/") -> pd.DataFrame:
    """選手名列を players/<slug>.html へのリンクに置き換える。"""
    out = df.copy()
    out[name_col] = [
        f'<a href="{prefix}{slugify(team, name)}.html">{name}</a>'
        for name, team in zip(out[name_col], out["球団"])
    ]
    return out


def colorize_teams(df: pd.DataFrame, team_col: str = "チーム") -> pd.DataFrame:
    """球団名セルをその球団カラーで着色し、チーム個別ページへのリンクにする。"""
    out = df.copy()
    out[team_col] = [
        f'<a href="team/{t}.html" style="color:{TEAM_COLORS.get(t, "inherit")}; font-weight:700">{t}</a>'
        for t in out[team_col]
    ]
    return out


def _fmt_int(v) -> str:
    return str(int(round(float(v))))


def _fmt_fixed(decimals: int):
    def _f(v) -> str:
        return f"{float(v):.{decimals}f}"
    return _f


def _fmt_pct1(v) -> str:
    return f"{float(v):.1f}%"


_INT_COLS = [
    "試合", "打席", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打", "塁打", "打点", "盗塁", "盗塁刺",
    "犠打", "犠飛", "四球", "故意四", "死球", "三振", "併殺打", "登板", "勝利", "敗北", "セーブ", "ホールド",
    "ＨＰ", "完投", "完封勝", "無四球", "打者", "暴投", "ボーク", "失点", "自責点",
    "直近試合数", "直近打数", "直近安打", "直近打点", "直近奪三振",
]
_FIXED3_COLS = [
    "打率", "長打率", "出塁率", "勝率", "ピタゴラス勝率", "OPS", "ISO", "wOBA", "BABIP",
    "回帰後打率", "回帰後出塁率", "回帰後長打率", "予想最終OPS",
    "予想最終打率", "予想最終出塁率", "予想最終長打率",
    "直近打率", "シーズン打率", "打率差",
]
_FIXED2_COLS = [
    "防御率", "WHIP", "FIP", "K/9", "BB/9", "K/BB", "RC27", "回帰後防御率", "回帰後WHIP",
    "予想最終防御率", "予想最終WHIP", "WAR",
    "直近防御率", "シーズン防御率", "防御率差",
]
_FIXED1_COLS = [
    "RC", "予想最終本塁打数", "期待勝利数", "予想最終勝利数(平均)", "運(実勝利-期待勝利)", "得失点差",
    "予想最終投球回", "予想最終奪三振数", "予想最終勝利数", "予想最終敗戦数",
    "直近投球回",
]

STAT_FORMAT = {}
STAT_FORMAT.update({c: _fmt_int for c in _INT_COLS})
STAT_FORMAT.update({c: _fmt_fixed(3) for c in _FIXED3_COLS})
STAT_FORMAT.update({c: _fmt_fixed(2) for c in _FIXED2_COLS})
STAT_FORMAT.update({c: _fmt_fixed(1) for c in _FIXED1_COLS})
STAT_FORMAT.update({c: _fmt_pct1 for c in ("K%", "BB%")})
STAT_FORMAT["wRC+"] = _fmt_int


def kv_table(row: pd.Series, columns: list[str], labels: dict[str, str] | None = None) -> str:
    """1選手・1チーム分のスタッツを、項目ごとの小さなタイルを並べた複数列グリッドにする。"""
    labels = labels or {}
    items = []
    for c in columns:
        v = row[c]
        formatter = STAT_FORMAT.get(c)
        v = formatter(v) if formatter else str(v)
        label = term(c, labels.get(c, c))
        items.append(f'<div class="kv-item"><div class="kv-label">{label}</div><div class="kv-value">{v}</div></div>')
    return '<div class="kv-grid">' + "".join(items) + "</div>"


def make_batting_chart(df: pd.DataFrame, league_label: str) -> str:
    top = df.nlargest(15, "OPS").sort_values("OPS")
    colors = [TEAM_COLORS.get(t, "#3ea6ff") for t in top["球団"]]
    fig = go.Figure(go.Bar(
        x=top["OPS"], y=top["選手"] + "(" + top["球団"] + ")", orientation="h",
        marker_color=colors, text=top["OPS"].map(lambda v: f"{v:.3f}"), textposition="outside",
    ))
    fig.update_layout(title=f"{league_label} OPSランキング Top15")
    fig.update_xaxes(title="OPS")
    return style_fig(fig, height=460)


def make_war_chart(df: pd.DataFrame, league_label: str) -> str:
    top = df.nlargest(15, "WAR").sort_values("WAR")
    colors = [TEAM_COLORS.get(t, "#3ea6ff") for t in top["球団"]]
    fig = go.Figure(go.Bar(
        x=top["WAR"], y=top["選手"] + "(" + top["球団"] + ")", orientation="h",
        marker_color=colors, text=top["WAR"].map(lambda v: f"{v:.2f}"), textposition="outside",
    ))
    fig.update_layout(title=f"{league_label} WAR(簡易版)ランキング Top15")
    fig.update_xaxes(title="WAR")
    return style_fig(fig, height=460)


def make_pitching_chart(df: pd.DataFrame, league_label: str) -> str:
    top = df.nsmallest(15, "FIP").sort_values("FIP", ascending=False)
    colors = [TEAM_COLORS.get(t, "#3ea6ff") for t in top["球団"]]
    fig = go.Figure(go.Bar(
        x=top["FIP"], y=top["投手"] + "(" + top["球団"] + ")", orientation="h",
        marker_color=colors, text=top["FIP"].map(lambda v: f"{v:.2f}"), textposition="outside",
    ))
    fig.update_layout(title=f"{league_label} FIPランキング Top15 (低いほど良い)")
    fig.update_xaxes(title="FIP")
    return style_fig(fig, height=460)


def make_runs_chart(team_saber: pd.DataFrame) -> str:
    fig = go.Figure()
    for lg, marker in (("セ", "circle"), ("パ", "diamond")):
        sub = team_saber[team_saber["リーグ"] == lg]
        fig.add_trace(go.Scatter(
            x=sub["得点"], y=sub["失点"], mode="markers+text", text=sub["チーム"],
            textposition="top center", name=f"{lg}・リーグ",
            marker=dict(size=13, color=[TEAM_COLORS.get(t, "#3ea6ff") for t in sub["チーム"]], symbol=marker,
                        line=dict(width=1, color="#ffffff")),
        ))
    lo = min(team_saber["得点"].min(), team_saber["失点"].min()) - 20
    hi = max(team_saber["得点"].max(), team_saber["失点"].max()) + 20
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="#aaaaaa"), showlegend=False))
    fig.update_layout(title="得点 vs 失点 (右下ほど得失点差が良い)")
    fig.update_xaxes(title="得点")
    fig.update_yaxes(title="失点")
    return style_fig(fig, height=520)


def make_luck_chart(team_saber: pd.DataFrame) -> str:
    sub = team_saber.sort_values("運(実勝利-期待勝利)")
    colors = ["#2f9e5c" if v >= 0 else "#cc3f57" for v in sub["運(実勝利-期待勝利)"]]
    fig = go.Figure(go.Bar(x=sub["運(実勝利-期待勝利)"], y=sub["チーム"], orientation="h", marker_color=colors))
    fig.update_layout(title="実勝利数 − ピタゴラス期待勝利数")
    fig.update_xaxes(title="差(勝)")
    return style_fig(fig, height=460)


def make_sim_chart(sim_df: pd.DataFrame, league_label: str) -> str:
    sub = sim_df.sort_values("優勝確率")
    colors = [TEAM_COLORS.get(t, "#3ea6ff") for t in sub["チーム"]]
    fig = go.Figure(go.Bar(x=sub["優勝確率"] * 100, y=sub["チーム"], orientation="h", marker_color=colors,
                            text=sub["優勝確率"].map(pct), textposition="outside"))
    fig.update_layout(title=f"{league_label} 優勝確率 (モンテカルロ・シミュレーション)")
    fig.update_xaxes(title="優勝確率 (%)")
    return style_fig(fig, height=340)


def _form_dots_html(results: list[str]) -> str:
    return "".join(f'<span class="form-dot {r}">{r}</span>' for r in results)


def make_matchup_card(
    home_team: str, away_team: str, home_pitcher: str | None, away_pitcher: str | None, info: str,
    home_prob: float, away_prob: float,
    home_fip: float, home_fip_known: bool, away_fip: float, away_fip_known: bool,
    home_form: dict, away_form: dict,
) -> str:
    home_color = TEAM_COLORS.get(home_team, "#3ea6ff")
    away_color = TEAM_COLORS.get(away_team, "#3ea6ff")
    home_pitcher_label = home_pitcher or "未定"
    away_pitcher_label = away_pitcher or "未定"
    home_fip_note = "" if home_fip_known else " (チーム平均)"
    away_fip_note = "" if away_fip_known else " (チーム平均)"

    return f"""
<div class="matchup-card">
  <div class="matchup-header">
    <div class="teams"><a href="team/{home_team}.html" style="color:{home_color}">{home_team}</a> vs
      <a href="team/{away_team}.html" style="color:{away_color}">{away_team}</a></div>
    <div class="info">{info}</div>
  </div>
  <div class="matchup-prob-bar">
    <div style="flex-grow:{home_prob:.4f}; flex-basis:0; background:{home_color}">{home_team} {home_prob * 100:.0f}%</div>
    <div style="flex-grow:{away_prob:.4f}; flex-basis:0; background:{away_color}">{away_team} {away_prob * 100:.0f}%</div>
  </div>
  <div class="matchup-grid">
    <div class="matchup-side">
      <div class="team-name" style="color:{home_color}">{home_team} 予告先発</div>
      <div class="pitcher-line">{home_pitcher_label} <span class="note">FIP {home_fip:.2f}{home_fip_note}</span></div>
      <div class="form-line">直近5試合 {home_form['record']} (得点{home_form['runs_for']:.0f}-失点{home_form['runs_against']:.0f})<span class="form-dots">{_form_dots_html(home_form['results'])}</span></div>
    </div>
    <div class="matchup-side">
      <div class="team-name" style="color:{away_color}">{away_team} 予告先発</div>
      <div class="pitcher-line">{away_pitcher_label} <span class="note">FIP {away_fip:.2f}{away_fip_note}</span></div>
      <div class="form-line">直近5試合 {away_form['record']} (得点{away_form['runs_for']:.0f}-失点{away_form['runs_against']:.0f})<span class="form-dots">{_form_dots_html(away_form['results'])}</span></div>
    </div>
  </div>
</div>
"""


def make_log5_heatmap(standings: pd.DataFrame, teams: list[str], league_label: str) -> str:
    win_pct = standings.set_index("チーム")["勝率"]
    matrix = [[predictions.log5_win_prob(win_pct[row], win_pct[col]) for col in teams] for row in teams]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=teams, y=teams, colorscale="RdBu", zmid=0.5, zmin=0.25, zmax=0.75,
        text=[[f"{v * 100:.0f}%" for v in row] for row in matrix], texttemplate="%{text}",
        colorbar=dict(title="勝率"), xgap=2, ygap=2,
    ))
    fig.update_layout(title=f"{league_label} 対戦カード勝率 (log5法・行が勝つ確率)")
    style_fig(fig, height=460)
    fig.update_xaxes(automargin=True, side="bottom", tickangle=0)
    fig.update_yaxes(automargin=True, autorange="reversed")
    fig.update_layout(margin=dict(l=90, r=20, t=50, b=70))
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)


def _to_num(v) -> float:
    v = (v or "").strip()
    if v in ("", "-", "--"):
        return 0.0
    return float(v)


def fetch_unqualified_players(
    year: int, qualified_bat_keys: set, qualified_pit_keys: set
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """規定打席・規定投球回に達していない支配下選手の、今季の実成績を取得する。

    各球団の選手名鑑を取得し、既に規定到達リストに載っている選手を除いたうえで
    個人ページから今季の成績を取得する。初回実行時は選手数が多く時間がかかるが、
    取得結果はキャッシュされるため2回目以降は高速。
    あわせて全選手(規定到達済みも含む)の守備位置・背番号・個人ページIDを
    player_info_lookup として返す(簡易版WARの守備位置補正、顔写真URLの組み立てに使う)。
    """
    batting_rows = []
    pitching_rows = []
    player_info_lookup: dict[tuple[str, str], dict] = {}
    for team in ALL_TEAMS:
        league = "セ" if team in CENTRAL_TEAMS else "パ"
        roster = scraper.fetch_roster(team)
        for _, p in roster.iterrows():
            player_info_lookup[(team, p["name"])] = {
                "position": p["position"], "number": p["number"], "player_id": p["player_id"],
            }
        roster = roster[roster["category"].str.contains("支配下", na=False)]
        for _, p in roster.iterrows():
            name = p["name"]
            need_batting = (team, name) not in qualified_bat_keys
            need_pitching = (team, name) not in qualified_pit_keys
            if not need_batting and not need_pitching:
                continue
            try:
                season = scraper.fetch_player_season(p["player_id"], year)
            except Exception as e:
                print(f"  ! {team} {name} の成績取得に失敗、スキップします ({e})")
                continue

            if need_batting and season["batting"]:
                b = season["batting"]
                pa = _to_num(b.get("打席"))
                if pa > 0:
                    batting_rows.append({
                        "選手": name, "球団": team, "年度": year, "リーグ": league,
                        "打率": _to_num(b.get("打率")), "試合": _to_num(b.get("試合")), "打席": pa,
                        "打数": _to_num(b.get("打数")), "得点": _to_num(b.get("得点")), "安打": _to_num(b.get("安打")),
                        "二塁打": _to_num(b.get("二塁打")), "三塁打": _to_num(b.get("三塁打")), "本塁打": _to_num(b.get("本塁打")),
                        "塁打": _to_num(b.get("塁打")), "打点": _to_num(b.get("打点")), "盗塁": _to_num(b.get("盗塁")),
                        "盗塁刺": _to_num(b.get("盗塁刺")), "犠打": _to_num(b.get("犠打")), "犠飛": _to_num(b.get("犠飛")),
                        "四球": _to_num(b.get("四球")), "故意四": _to_num(b.get("故意四")), "死球": _to_num(b.get("死球")),
                        "三振": _to_num(b.get("三振")), "併殺打": _to_num(b.get("併殺打")),
                        "長打率": _to_num(b.get("長打率")), "出塁率": _to_num(b.get("出塁率")),
                    })

            if need_pitching and season["pitching"]:
                pt = season["pitching"]
                ip = sabermetrics.innings_to_float(pt.get("投球回", "0.0"))
                if ip > 0:
                    pitching_rows.append({
                        "投手": name, "球団": team, "年度": year, "リーグ": league,
                        "防御率": _to_num(pt.get("防御率")), "登板": _to_num(pt.get("登板")), "勝利": _to_num(pt.get("勝利")),
                        "敗北": _to_num(pt.get("敗北")), "セーブ": _to_num(pt.get("セーブ")), "ホールド": _to_num(pt.get("ホールド")),
                        "ＨＰ": _to_num(pt.get("ＨＰ")), "完投": _to_num(pt.get("完投")), "完封勝": _to_num(pt.get("完封勝")),
                        "無四球": _to_num(pt.get("無四球")), "勝率": _to_num(pt.get("勝率")), "打者": _to_num(pt.get("打者")),
                        "投球回": pt.get("投球回", "0.0"), "安打": _to_num(pt.get("安打")), "本塁打": _to_num(pt.get("本塁打")),
                        "四球": _to_num(pt.get("四球")), "故意四": _to_num(pt.get("故意四")), "死球": _to_num(pt.get("死球")),
                        "三振": _to_num(pt.get("三振")), "暴投": _to_num(pt.get("暴投")), "ボーク": _to_num(pt.get("ボーク")),
                        "失点": _to_num(pt.get("失点")), "自責点": _to_num(pt.get("自責点")),
                    })

    batting_df = pd.DataFrame(batting_rows)
    pitching_df = pd.DataFrame(pitching_rows)

    bat_int_cols = ["試合", "打席", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打", "塁打", "打点",
                    "盗塁", "盗塁刺", "犠打", "犠飛", "四球", "故意四", "死球", "三振", "併殺打"]
    pit_int_cols = ["登板", "勝利", "敗北", "セーブ", "ホールド", "ＨＰ", "完投", "完封勝", "無四球", "打者",
                     "安打", "本塁打", "四球", "故意四", "死球", "三振", "暴投", "ボーク", "失点", "自責点"]
    if not batting_df.empty:
        batting_df[bat_int_cols] = batting_df[bat_int_cols].astype(int)
    if not pitching_df.empty:
        pitching_df[pit_int_cols] = pitching_df[pit_int_cols].astype(int)

    return batting_df, pitching_df, player_info_lookup


def compute_recent_trends(
    game_log: pd.DataFrame,
    all_batters_full: pd.DataFrame,
    all_pitchers_full: pd.DataFrame,
    id_to_player: dict,
    *,
    n_games: int = 10,
    min_ab: int = 15,
    min_ip: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """直近n_games試合のボックススコアを集計し、シーズン成績との差分(好調/不調)を計算する。"""
    completed = game_log[game_log["completed"] & game_log["game_url"].notna()]

    team_recent_rows = {}
    needed_urls: set[str] = set()
    for team in ALL_TEAMS:
        mask = (completed["team1"] == team) | (completed["team2"] == team)
        recent = completed[mask].tail(n_games)
        team_recent_rows[team] = recent
        needed_urls.update(recent["game_url"].tolist())

    box_cache = {}
    for url in needed_urls:
        try:
            box_cache[url] = scraper.fetch_box_score(url)
        except Exception as e:
            print(f"  ! ボックススコア取得に失敗、スキップ: {url} ({e})")

    trends_bat = []
    trends_pit = []
    for team in ALL_TEAMS:
        bat_agg: dict[str, dict] = {}
        pit_agg: dict[str, dict] = {}
        for _, g in team_recent_rows[team].iterrows():
            box = box_cache.get(g["game_url"])
            if not box:
                continue
            # ボックススコアの team1/team2 (表/裏打撃) は試合日程の team1/team2 と順序が逆になっている。
            side = "team2" if g["team1"] == team else "team1"
            for row in box[f"{side}_batting"]:
                a = bat_agg.setdefault(row["player_id"], {"AB": 0, "H": 0, "RBI": 0, "G": 0})
                a["AB"] += row["AB"]
                a["H"] += row["H"]
                a["RBI"] += row["RBI"]
                a["G"] += 1
            for row in box[f"{side}_pitching"]:
                p = pit_agg.setdefault(row["player_id"], {"IP": 0.0, "ER": 0, "SO": 0, "G": 0})
                p["IP"] += sabermetrics.innings_to_float(row["IP"])
                p["ER"] += row["ER"]
                p["SO"] += row["SO"]
                p["G"] += 1

        for pid, a in bat_agg.items():
            key = id_to_player.get(pid)
            if not key or a["AB"] < min_ab:
                continue
            team_, name_ = key
            trends_bat.append({
                "選手": name_, "球団": team_, "直近試合数": a["G"], "直近打数": a["AB"],
                "直近安打": a["H"], "直近打点": a["RBI"], "直近打率": round(a["H"] / a["AB"], 3),
            })
        for pid, p in pit_agg.items():
            key = id_to_player.get(pid)
            if not key or p["IP"] < min_ip:
                continue
            team_, name_ = key
            trends_pit.append({
                "投手": name_, "球団": team_, "直近試合数": p["G"], "直近投球回": round(p["IP"], 1),
                "直近奪三振": p["SO"], "直近防御率": round(p["ER"] * 9 / p["IP"], 2),
            })

    bat_df = pd.DataFrame(trends_bat)
    pit_df = pd.DataFrame(trends_pit)

    season_avg = all_batters_full.set_index(["球団", "選手"])["打率"]
    if not bat_df.empty:
        bat_df["シーズン打率"] = [season_avg.get((t, n), float("nan")) for t, n in zip(bat_df["球団"], bat_df["選手"])]
        bat_df = bat_df.dropna(subset=["シーズン打率"]).copy()
        bat_df["打率差"] = (bat_df["直近打率"] - bat_df["シーズン打率"]).round(3)

    season_era = all_pitchers_full.set_index(["球団", "投手"])["防御率"]
    if not pit_df.empty:
        pit_df["シーズン防御率"] = [season_era.get((t, n), float("nan")) for t, n in zip(pit_df["球団"], pit_df["投手"])]
        pit_df = pit_df.dropna(subset=["シーズン防御率"]).copy()
        pit_df["防御率差"] = (pit_df["シーズン防御率"] - pit_df["直近防御率"]).round(2)

    return bat_df, pit_df


def main() -> None:
    print(f"[1/10] {YEAR}年度データを取得中 (npb.jp)...")
    data = scraper.fetch_all(YEAR)
    batting, pitching = data["batting"], data["pitching"]
    team_batting, team_pitching = data["team_batting"], data["team_pitching"]
    standings = data["standings"]

    print("[2/10] セイバーメトリクスを算出中...")
    batting_saber = {}
    pitching_saber = {}
    fip_const_by_league = {}
    for lg_label, lg_code in (("セ", "セ"), ("パ", "パ")):
        bat_lg = sabermetrics.add_batting_sabermetrics(batting[batting["リーグ"] == lg_code])
        batting_saber[lg_code] = bat_lg.sort_values("OPS", ascending=False)

        tp_lg = team_pitching[team_pitching["チーム"].isin(CENTRAL_TEAMS if lg_code == "セ" else PACIFIC_TEAMS)]
        fip_const = sabermetrics.league_fip_constant(tp_lg)
        fip_const_by_league[lg_code] = fip_const
        pit_lg = sabermetrics.add_pitching_sabermetrics(pitching[pitching["リーグ"] == lg_code], fip_const)
        pitching_saber[lg_code] = pit_lg.sort_values("FIP")

    team_saber = sabermetrics.add_team_sabermetrics(standings, team_batting, team_pitching)

    print("[3/10] 規定未達選手のロースター・成績を取得中 (初回は数分かかります)...")
    qualified_bat_keys = {(r["球団"], r["選手"]) for df in batting_saber.values() for _, r in df.iterrows()}
    qualified_pit_keys = {(r["球団"], r["投手"]) for df in pitching_saber.values() for _, r in df.iterrows()}
    unq_batting_raw, unq_pitching_raw, player_info_lookup = fetch_unqualified_players(
        YEAR, qualified_bat_keys, qualified_pit_keys
    )

    def _add_position_and_war(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["position"] = [player_info_lookup.get((t, n), {}).get("position", "") for t, n in zip(df["球団"], df[name_col])]
        return sabermetrics.add_war(df)

    unq_bat_saber = {}
    unq_pit_saber = {}
    for lg_code in ("セ", "パ"):
        bat_lg = batting_saber[lg_code]
        batting_saber[lg_code] = _add_position_and_war(bat_lg, "選手")

        b = unq_batting_raw[unq_batting_raw["リーグ"] == lg_code]
        unq_bat_saber[lg_code] = sabermetrics.add_batting_sabermetrics(b) if not b.empty else b
        unq_bat_saber[lg_code] = _add_position_and_war(unq_bat_saber[lg_code], "選手")
        p = unq_pitching_raw[unq_pitching_raw["リーグ"] == lg_code]
        unq_pit_saber[lg_code] = sabermetrics.add_pitching_sabermetrics(p, fip_const_by_league[lg_code]) if not p.empty else p
    print(f"  -> 規定未達: 打者{len(unq_batting_raw)}名 / 投手{len(unq_pitching_raw)}名")

    print("[4/10] 順位・勝敗予想を計算中...")
    sim = {}
    for lg_code, teams in (("セ", CENTRAL_TEAMS), ("パ", PACIFIC_TEAMS)):
        sub = team_saber[team_saber["リーグ"] == lg_code]
        sim[lg_code] = predictions.simulate_final_standings(sub)

    all_batters_all_league = pd.concat(batting_saber.values(), ignore_index=True)
    proj_batters = predictions.project_batter_rest_of_season(all_batters_all_league, team_batting, standings)
    proj_batters_top = proj_batters.nlargest(15, "予想最終OPS")

    all_batters_full = pd.concat([all_batters_all_league] + list(unq_bat_saber.values()), ignore_index=True)
    proj_batters_full = predictions.project_batter_rest_of_season(all_batters_full, team_batting, standings)

    all_pitchers_all_league = pd.concat(pitching_saber.values(), ignore_index=True)
    all_pitchers_full = pd.concat([all_pitchers_all_league] + list(unq_pit_saber.values()), ignore_index=True)
    proj_pitchers_full = predictions.project_pitcher_rest_of_season(all_pitchers_full, team_pitching, standings)

    print("[5/10] HTML・グラフを生成中...")
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    env.globals["term"] = term
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    page_titles = {
        "index.html": "概要",
        "batting.html": "打撃成績",
        "pitching.html": "投手成績",
        "teams.html": "チーム成績",
        "predictions.html": "順位・勝敗予想",
        "tomorrow.html": "次の対戦カード予想",
        "trend.html": "直近10試合の好調・不調",
    }

    def render(name: str, active: str, **ctx):
        tpl = env.get_template(name)
        html = tpl.render(active=active, year=YEAR, generated_at=generated_at, page_title=page_titles[name], **ctx)
        (SITE_DIR / name).write_text(html, encoding="utf-8")

    def render_to(out_path: Path, template_name: str, *, active: str, page_title: str, path_prefix: str, **ctx):
        tpl = env.get_template(template_name)
        html = tpl.render(
            active=active, year=YEAR, generated_at=generated_at, page_title=page_title,
            path_prefix=path_prefix, **ctx,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

    # --- index ---
    standings_display_cols = ["チーム", "試合", "勝利", "敗北", "引分", "勝率", "差", "得失点差", "ピタゴラス勝率", "運(実勝利-期待勝利)"]
    leader_c = team_saber[team_saber["リーグ"] == "セ"].nlargest(1, "勝率")["チーム"].iloc[0]
    leader_p = team_saber[team_saber["リーグ"] == "パ"].nlargest(1, "勝率")["チーム"].iloc[0]
    top_ops_player = all_batters_all_league.nlargest(1, "OPS").iloc[0]
    headline_tiles = [
        {"label": "セ・リーグ首位", "value": leader_c},
        {"label": "パ・リーグ首位", "value": leader_p},
        {"label": "OPS 1位", "value": f"{top_ops_player['選手']} ({top_ops_player['OPS']:.3f})"},
    ]
    render(
        "index.html", "index",
        headline_tiles=headline_tiles,
        standings_c=df_to_html(colorize_teams(team_saber[team_saber["リーグ"] == "セ"][standings_display_cols].sort_values("勝率", ascending=False))),
        standings_p=df_to_html(colorize_teams(team_saber[team_saber["リーグ"] == "パ"][standings_display_cols].sort_values("勝率", ascending=False))),
    )

    # --- batting ---
    bat_cols = ["選手", "球団", "打率", "打席", "本塁打", "打点", "出塁率", "長打率", "OPS", "ISO", "wOBA", "RC27", "BABIP", "K%", "BB%", "wRC+", "WAR"]
    render(
        "batting.html", "batting",
        chart_bat_c=make_batting_chart(batting_saber["セ"], "セ・リーグ"),
        chart_bat_p=make_batting_chart(batting_saber["パ"], "パ・リーグ"),
        chart_war_c=make_war_chart(batting_saber["セ"], "セ・リーグ"),
        chart_war_p=make_war_chart(batting_saber["パ"], "パ・リーグ"),
        table_bat_c=df_to_html(colorize_teams(linkify_players(batting_saber["セ"][bat_cols], "選手"), "球団")),
        table_bat_p=df_to_html(colorize_teams(linkify_players(batting_saber["パ"][bat_cols], "選手"), "球団")),
    )

    # --- pitching ---
    pit_cols = ["投手", "球団", "防御率", "登板", "勝利", "敗北", "セーブ", "投球回", "WHIP", "FIP", "K/9", "BB/9", "K/BB", "K%", "BB%"]
    render(
        "pitching.html", "pitching",
        chart_pit_c=make_pitching_chart(pitching_saber["セ"], "セ・リーグ"),
        chart_pit_p=make_pitching_chart(pitching_saber["パ"], "パ・リーグ"),
        table_pit_c=df_to_html(colorize_teams(linkify_players(pitching_saber["セ"][pit_cols], "投手"), "球団")),
        table_pit_p=df_to_html(colorize_teams(linkify_players(pitching_saber["パ"][pit_cols], "投手"), "球団")),
    )

    # --- teams ---
    team_cols = ["チーム", "試合", "勝利", "敗北", "引分", "勝率", "得点", "失点", "得失点差", "ピタゴラス勝率", "期待勝利数", "運(実勝利-期待勝利)"]
    render(
        "teams.html", "teams",
        chart_runs=make_runs_chart(team_saber),
        chart_luck=make_luck_chart(team_saber),
        table_teams_c=df_to_html(colorize_teams(team_saber[team_saber["リーグ"] == "セ"][team_cols].sort_values("勝率", ascending=False))),
        table_teams_p=df_to_html(colorize_teams(team_saber[team_saber["リーグ"] == "パ"][team_cols].sort_values("勝率", ascending=False))),
    )

    # --- predictions ---
    sim_cols = ["チーム", "試合", "勝利", "残り試合", "ピタゴラス勝率", "予想最終勝利数(平均)", "優勝確率", "CS進出確率(上位3位)"]
    sim_c_fmt = sim["セ"].copy()
    sim_p_fmt = sim["パ"].copy()
    for d in (sim_c_fmt, sim_p_fmt):
        d["優勝確率"] = d["優勝確率"].map(pct)
        d["CS進出確率(上位3位)"] = d["CS進出確率(上位3位)"].map(pct)

    proj_cols = ["選手", "球団", "打席", "予想最終打率", "予想最終出塁率", "予想最終長打率", "予想最終OPS", "本塁打", "予想最終本塁打数"]
    render(
        "predictions.html", "predictions",
        scheduled_games=predictions.SCHEDULED_GAMES,
        n_sim=20000,
        table_sim_c=df_to_html(colorize_teams(sim_c_fmt[sim_cols])),
        table_sim_p=df_to_html(colorize_teams(sim_p_fmt[sim_cols])),
        chart_sim_c=make_sim_chart(sim["セ"], "セ・リーグ"),
        chart_sim_p=make_sim_chart(sim["パ"], "パ・リーグ"),
        chart_log5_c=make_log5_heatmap(standings[standings["リーグ"] == "セ"], CENTRAL_TEAMS, "セ・リーグ"),
        chart_log5_p=make_log5_heatmap(standings[standings["リーグ"] == "パ"], PACIFIC_TEAMS, "パ・リーグ"),
        table_proj_batters=df_to_html(colorize_teams(linkify_players(proj_batters_top[proj_cols], "選手"), "球団")),
    )

    print("[6/10] 選手個別ページを生成中...")
    players_dir = SITE_DIR / "players"
    if players_dir.exists():
        shutil.rmtree(players_dir)

    players: dict[tuple[str, str], dict] = {}
    for lg, df in batting_saber.items():
        for _, row in df.iterrows():
            key = (row["球団"], row["選手"])
            players.setdefault(key, {"league": lg})["batting"] = row
            players[key]["batting_qualified"] = True
    for lg, df in pitching_saber.items():
        for _, row in df.iterrows():
            key = (row["球団"], row["投手"])
            players.setdefault(key, {"league": lg})["pitching"] = row
            players[key]["pitching_qualified"] = True
    for lg, df in unq_bat_saber.items():
        for _, row in df.iterrows():
            key = (row["球団"], row["選手"])
            players.setdefault(key, {"league": lg})["batting"] = row
            players[key].setdefault("batting_qualified", False)
    for lg, df in unq_pit_saber.items():
        for _, row in df.iterrows():
            key = (row["球団"], row["投手"])
            players.setdefault(key, {"league": lg})["pitching"] = row
            players[key].setdefault("pitching_qualified", False)

    proj_index = proj_batters_full.set_index(["球団", "選手"])
    proj_pitch_index = proj_pitchers_full.set_index(["球団", "投手"])
    bat_kv_cols = ["打率", "試合", "打席", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打", "塁打", "打点",
                   "盗塁", "盗塁刺", "犠打", "犠飛", "四球", "故意四", "死球", "三振", "併殺打", "長打率", "出塁率",
                   "OPS", "ISO", "wOBA", "RC", "RC27", "BABIP", "K%", "BB%", "wRC+", "WAR"]
    pit_kv_cols = ["防御率", "登板", "勝利", "敗北", "セーブ", "ホールド", "ＨＰ", "完投", "完封勝", "無四球", "勝率",
                   "打者", "投球回", "安打", "本塁打", "四球", "故意四", "死球", "三振", "暴投", "ボーク", "失点",
                   "自責点", "WHIP", "FIP", "K/9", "BB/9", "K/BB", "K%", "BB%"]

    for (team, name), info in players.items():
        p_info = player_info_lookup.get((team, name), {})
        photo_url = scraper.player_photo_url(team, p_info.get("number", ""), p_info.get("player_id", ""), YEAR)
        ctx = {
            "name": name, "team": team, "league": info["league"], "team_color": TEAM_COLORS.get(team, "inherit"),
            "photo_url": photo_url, "team_logo_url": scraper.team_logo_url(team, YEAR),
        }

        if "batting" in info:
            r = info["batting"]
            ctx["batting_tiles"] = [
                {"label": "OPS", "value": f"{r['OPS']:.3f}"},
                {"label": "打率", "value": f"{r['打率']:.3f}"},
                {"label": "本塁打", "value": f"{int(r['本塁打'])}"},
                {"label": "打点", "value": f"{int(r['打点'])}"},
                {"label": "出塁率", "value": f"{r['出塁率']:.3f}"},
                {"label": "長打率", "value": f"{r['長打率']:.3f}"},
                {"label": "WAR(簡易)", "value": f"{r['WAR']:.2f}"},
                {"label": "wRC+", "value": f"{int(r['wRC+'])}"},
            ]
            ctx["batting_table"] = kv_table(r, bat_kv_cols)
            ctx["batting_qualified"] = info.get("batting_qualified", False)
            ctx["list_page"] = "batting.html"
            ctx["list_label"] = "打撃成績"

            if (team, name) in proj_index.index:
                prow = proj_index.loc[(team, name)]
                if isinstance(prow, pd.DataFrame):
                    prow = prow.iloc[0]
                ctx["projection_table"] = kv_table(
                    prow,
                    ["打席", "予想最終打率", "予想最終出塁率", "予想最終長打率", "予想最終OPS", "本塁打", "予想最終本塁打数"],
                    labels={
                        "打席": "現在の打席数",
                        "予想最終打率": "シーズン最終予想打率",
                        "予想最終出塁率": "シーズン最終予想出塁率",
                        "予想最終長打率": "シーズン最終予想長打率",
                        "予想最終OPS": "シーズン最終予想OPS",
                        "本塁打": "現在の本塁打数",
                        "予想最終本塁打数": "シーズン最終予想本塁打数",
                    },
                )

        if "pitching" in info:
            r = info["pitching"]
            ctx["pitching_tiles"] = [
                {"label": "防御率", "value": f"{r['防御率']:.2f}"},
                {"label": "勝敗", "value": f"{int(r['勝利'])}勝{int(r['敗北'])}敗"},
                {"label": "奪三振", "value": f"{int(r['三振'])}"},
                {"label": "WHIP", "value": f"{r['WHIP']:.2f}"},
                {"label": "FIP", "value": f"{r['FIP']:.2f}"},
                {"label": "K/9", "value": f"{r['K/9']:.2f}"},
            ]
            ctx["pitching_table"] = kv_table(r, pit_kv_cols)
            ctx["pitching_qualified"] = info.get("pitching_qualified", False)
            ctx.setdefault("list_page", "pitching.html")
            ctx.setdefault("list_label", "投手成績")

            if (team, name) in proj_pitch_index.index:
                prow = proj_pitch_index.loc[(team, name)]
                if isinstance(prow, pd.DataFrame):
                    prow = prow.iloc[0]
                ctx["pitching_projection_table"] = kv_table(
                    prow,
                    ["予想最終防御率", "予想最終WHIP", "予想最終投球回", "予想最終奪三振数", "予想最終勝利数", "予想最終敗戦数"],
                    labels={
                        "予想最終防御率": "シーズン最終予想防御率",
                        "予想最終WHIP": "シーズン最終予想WHIP",
                        "予想最終投球回": "シーズン最終予想投球回",
                        "予想最終奪三振数": "シーズン最終予想奪三振数",
                        "予想最終勝利数": "シーズン最終予想勝利数",
                        "予想最終敗戦数": "シーズン最終予想敗戦数",
                    },
                )

        out_path = players_dir / f"{slugify(team, name)}.html"
        render_to(
            out_path, "player.html", active="", page_title=f"{name}（{team}）",
            path_prefix="../", **ctx,
        )
    print(f"  -> {len(players)}選手分のページを生成")

    print("[7/10] チーム個別ページを生成中...")
    teams_dir = SITE_DIR / "team"
    if teams_dir.exists():
        shutil.rmtree(teams_dir)

    team_batting_kv_cols = ["打率", "試合", "打席", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打", "塁打",
                             "打点", "盗塁", "盗塁刺", "犠打", "犠飛", "四球", "故意四", "死球", "三振", "併殺打",
                             "長打率", "出塁率"]
    team_pitching_kv_cols = ["防御率", "試合", "勝利", "敗北", "セーブ", "ホールド", "ＨＰ", "完投", "完封勝",
                              "無四球", "勝率", "打者", "投球回", "安打", "本塁打", "四球", "故意四", "死球",
                              "三振", "暴投", "ボーク", "失点", "自責点"]

    team_batting_idx = team_batting.set_index("チーム")
    team_pitching_idx = team_pitching.set_index("チーム")
    sim_all = pd.concat(sim.values(), ignore_index=True).set_index("チーム")

    for lg_code, teams in (("セ", CENTRAL_TEAMS), ("パ", PACIFIC_TEAMS)):
        ranked = team_saber[team_saber["リーグ"] == lg_code].sort_values("勝率", ascending=False).reset_index(drop=True)
        for i, row in ranked.iterrows():
            team = row["チーム"]
            sim_row = sim_all.loc[team]
            tiles = [
                {"label": "リーグ順位", "value": f"{i + 1}位"},
                {"label": "勝敗", "value": f"{int(row['勝利'])}勝{int(row['敗北'])}敗{int(row['引分'])}分"},
                {"label": "勝率", "value": f"{row['勝率']:.3f}"},
                {"label": "得失点差", "value": f"{int(row['得失点差']):+d}"},
                {"label": "ピタゴラス勝率", "value": f"{row['ピタゴラス勝率']:.3f}"},
                {"label": "優勝確率", "value": pct(sim_row["優勝確率"])},
                {"label": "CS進出確率", "value": pct(sim_row["CS進出確率(上位3位)"])},
            ]
            batting_table = kv_table(team_batting_idx.loc[team], team_batting_kv_cols)
            pitching_table = kv_table(team_pitching_idx.loc[team], team_pitching_kv_cols)

            roster_bat = batting_saber[lg_code][batting_saber[lg_code]["球団"] == team][bat_cols]
            roster_pit = pitching_saber[lg_code][pitching_saber[lg_code]["球団"] == team][pit_cols]
            roster_batting_html = (
                df_to_html(linkify_players(roster_bat, "選手", prefix="../players/").drop(columns="球団"))
                if not roster_bat.empty else None
            )
            roster_pitching_html = (
                df_to_html(linkify_players(roster_pit, "投手", prefix="../players/").drop(columns="球団"))
                if not roster_pit.empty else None
            )

            unq_bat_team = unq_bat_saber[lg_code]
            unq_bat_team = unq_bat_team[unq_bat_team["球団"] == team][bat_cols] if not unq_bat_team.empty else unq_bat_team
            unq_pit_team = unq_pit_saber[lg_code]
            unq_pit_team = unq_pit_team[unq_pit_team["球団"] == team][pit_cols] if not unq_pit_team.empty else unq_pit_team
            other_batting_html = (
                df_to_html(linkify_players(unq_bat_team.sort_values("OPS", ascending=False), "選手", prefix="../players/").drop(columns="球団"))
                if not unq_bat_team.empty else None
            )
            other_pitching_html = (
                df_to_html(linkify_players(unq_pit_team.sort_values("FIP"), "投手", prefix="../players/").drop(columns="球団"))
                if not unq_pit_team.empty else None
            )

            render_to(
                teams_dir / f"{team}.html", "team.html", active="teams", page_title=f"{team} チーム成績",
                path_prefix="../", team=team, team_color=TEAM_COLORS.get(team, "inherit"), league=lg_code,
                team_logo_url=scraper.team_logo_url(team, YEAR),
                tiles=tiles, batting_table=batting_table, pitching_table=pitching_table,
                roster_batting=roster_batting_html, roster_pitching=roster_pitching_html,
                other_batting=other_batting_html, other_pitching=other_pitching_html,
            )
    print(f"  -> {len(ALL_TEAMS)}チーム分のページを生成")

    print("[8/10] 次の対戦カード予想を生成中...")
    starters = scraper.fetch_probable_starters()

    now = datetime.now()
    months_to_fetch = {(now.year, now.month)}
    prev_month = now.month - 1 or 12
    prev_year = now.year if now.month > 1 else now.year - 1
    months_to_fetch.add((prev_year, prev_month))
    game_log = pd.concat(
        [scraper.fetch_schedule_month(y, m) for y, m in sorted(months_to_fetch)], ignore_index=True
    )

    fip_lookup = all_pitchers_full.set_index(["球団", "投手"])["FIP"]
    team_avg_fip = all_pitchers_all_league.groupby("球団")["FIP"].mean()
    league_avg_fip = all_pitchers_all_league["FIP"].mean()
    season_win_pct = team_saber.set_index("チーム")["勝率"]

    def _lookup_fip(name: str | None, team: str) -> tuple[float, bool]:
        if name and (team, name) in fip_lookup.index:
            v = fip_lookup.loc[(team, name)]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            return float(v), True
        return float(team_avg_fip.get(team, league_avg_fip)), False

    matchup_cards = []
    for m in starters["matchups"]:
        home, away = m["home_team"], m["away_team"]
        home_form = predictions.recent_form(home, game_log, n=5)
        away_form = predictions.recent_form(away, game_log, n=5)
        home_blended = predictions.blended_win_pct(season_win_pct[home], home_form["win_pct"])
        away_blended = predictions.blended_win_pct(season_win_pct[away], away_form["win_pct"])
        base_prob = predictions.log5_win_prob(home_blended, away_blended)

        home_fip, home_fip_known = _lookup_fip(m["home_pitcher"], home)
        away_fip, away_fip_known = _lookup_fip(m["away_pitcher"], away)
        adj = predictions.starter_fip_adjustment(home_fip, away_fip)

        home_prob = min(0.95, max(0.05, base_prob + adj))
        away_prob = 1 - home_prob

        matchup_cards.append(make_matchup_card(
            home, away, m["home_pitcher"], m["away_pitcher"], m["info"],
            home_prob, away_prob, home_fip, home_fip_known, away_fip, away_fip_known,
            home_form, away_form,
        ))

    render(
        "tomorrow.html", "tomorrow",
        date_label=starters["date_label"], matchup_cards=matchup_cards,
    )
    print(f"  -> {len(matchup_cards)}試合分の予想を生成")

    print("[9/10] 直近10試合の好調・不調選手を集計中...")
    id_to_player = {info["player_id"]: key for key, info in player_info_lookup.items() if info.get("player_id")}
    trend_bat, trend_pit = compute_recent_trends(game_log, all_batters_full, all_pitchers_full, id_to_player)

    bat_up_cols = ["選手", "球団", "直近試合数", "直近打数", "直近安打", "直近打点", "直近打率", "シーズン打率", "打率差"]
    pit_up_cols = ["投手", "球団", "直近試合数", "直近投球回", "直近奪三振", "直近防御率", "シーズン防御率", "防御率差"]
    if not trend_bat.empty:
        bat_hot = df_to_html(colorize_teams(linkify_players(trend_bat.nlargest(15, "打率差")[bat_up_cols], "選手"), "球団"))
        bat_cold = df_to_html(colorize_teams(linkify_players(trend_bat.nsmallest(15, "打率差")[bat_up_cols], "選手"), "球団"))
    else:
        bat_hot = bat_cold = None
    if not trend_pit.empty:
        pit_hot = df_to_html(colorize_teams(linkify_players(trend_pit.nlargest(15, "防御率差")[pit_up_cols], "投手"), "球団"))
        pit_cold = df_to_html(colorize_teams(linkify_players(trend_pit.nsmallest(15, "防御率差")[pit_up_cols], "投手"), "球団"))
    else:
        pit_hot = pit_cold = None

    render(
        "trend.html", "trend",
        n_games=10, table_bat_hot=bat_hot, table_bat_cold=bat_cold,
        table_pit_hot=pit_hot, table_pit_cold=pit_cold,
    )
    print(f"  -> 打者{len(trend_bat)}名 / 投手{len(trend_pit)}名のトレンドを算出")

    print("[10/10] 静的ファイルをコピー中...")
    static_out = SITE_DIR / "static"
    if static_out.exists():
        shutil.rmtree(static_out)
    shutil.copytree(STATIC_DIR, static_out)

    print(f"\n完了しました。 {SITE_DIR / 'index.html'} をブラウザで開いてください。")


if __name__ == "__main__":
    SITE_DIR.mkdir(exist_ok=True)
    main()
