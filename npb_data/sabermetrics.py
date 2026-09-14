"""公式サイトのカウンティングスタッツから、セイバーメトリクス指標を計算する。

参照: https://baseball-89.com/sabrmetrics-list/ の指標定義に準拠。
UZR/DRS/wSB/UBRは打球追跡データ・詳細な守備データが必要で公式サイトの基本成績からは
算出できないため対象外。wOBA/RCはMLB由来の簡易係数を使った近似値である旨を明記する。

WAR(打者)は https://baseballhub.app/methodology の簡易算出法を採用した近似値。
OPSから概算したwRC+・出場試合数・盗塁数・守備位置補正のみで計算する簡易式であり、
UZR等の詳細な守備データは使っていないため、真の意味でのWARとは異なる参考値である。
投手WARは参照元でも「今後追加予定」として未提供のため、本サイトでも対象外とする。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# wOBA線形加重 (Tom Tango版係数の簡易流用。NPB特有の再計算はしていない近似値)
WOBA_WEIGHTS = {"bb": 0.690, "hbp": 0.722, "single": 0.888, "double": 1.271, "triple": 1.616, "hr": 2.101}

# baseballhub.app方式の守備位置補正 (フル出場換算)。
# 公式サイトの選手名鑑では内野手が一塁手/二塁手/三塁手/遊撃手に細分化されておらず
# 「内野手」で一括りにされるため、二塁手・三塁手・遊撃手向けの+0.30を内野手全体に適用する
# (一塁手をやや過大評価する近似)。DHは名鑑上の守備位置と区別できないため未対応(0扱い)。
POSITION_WAR_ADJUSTMENT = {"捕手": 1.25, "内野手": 0.30, "外野手": -0.30, "投手": 0.0}
WAR_SCHEDULED_GAMES = 143


def innings_to_float(ip) -> float:
    """'163.1' (163回1/3) 形式の投球回を実数に変換する。.1=1/3, .2=2/3。"""
    s = str(ip).strip()
    if not s:
        return 0.0
    if "." not in s:
        return float(s)
    whole, frac, *_ = s.split(".")
    whole = float(whole or 0)
    frac = int(frac or 0)
    return whole + frac / 3.0 if frac in (1, 2) else whole + frac / 10.0


def add_batting_sabermetrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pa = df["打席"].astype(float)
    ab = df["打数"].astype(float)
    h = df["安打"].astype(float)
    doubles = df["二塁打"].astype(float)
    triples = df["三塁打"].astype(float)
    hr = df["本塁打"].astype(float)
    tb = df["塁打"].astype(float)
    bb = df["四球"].astype(float)
    ibb = df["故意四"].astype(float)
    hbp = df["死球"].astype(float)
    so = df["三振"].astype(float)
    sf = df["犠飛"].astype(float)
    sh = df["犠打"].astype(float)
    cs = df["盗塁刺"].astype(float)
    gidp = df["併殺打"].astype(float)
    obp = df["出塁率"].astype(float)
    slg = df["長打率"].astype(float)
    avg = df["打率"].astype(float)

    singles = h - doubles - triples - hr
    bb_unintentional = bb - ibb

    df["OPS"] = (obp + slg).round(3)
    df["ISO"] = (slg - avg).round(3)

    woba_num = (
        WOBA_WEIGHTS["bb"] * bb_unintentional
        + WOBA_WEIGHTS["hbp"] * hbp
        + WOBA_WEIGHTS["single"] * singles
        + WOBA_WEIGHTS["double"] * doubles
        + WOBA_WEIGHTS["triple"] * triples
        + WOBA_WEIGHTS["hr"] * hr
    )
    woba_den = ab + bb_unintentional + sf + hbp
    df["wOBA"] = (woba_num / woba_den.replace(0, np.nan)).round(3)

    babip_den = ab - so - hr + sf
    df["BABIP"] = ((h - hr) / babip_den.replace(0, np.nan)).round(3)

    df["K%"] = (so / pa.replace(0, np.nan) * 100).round(1)
    df["BB%"] = (bb / pa.replace(0, np.nan) * 100).round(1)

    rc = (h + bb) * tb / (ab + bb).replace(0, np.nan)
    outs = (ab - h + sh + sf + cs + gidp).clip(lower=1)
    df["RC"] = rc.round(1)
    df["RC27"] = (rc / outs * 27).round(2)

    return df


def wrc_plus(ops) -> pd.Series:
    """baseballhub.app方式の簡易wRC+。100 + (OPS-0.700)*220を30〜220にクリップする。
    リーグ・球場補正は行っていない簡易版。
    """
    val = 100 + (ops - 0.700) * 220
    return val.clip(lower=30, upper=220)


def add_war(df: pd.DataFrame, *, scheduled_games: int = WAR_SCHEDULED_GAMES) -> pd.DataFrame:
    """baseballhub.app方式の簡易打者WARを算出する。事前に df に "position" 列
    (捕手/内野手/外野手/投手のいずれか) が必要。

    WAR = battingWAR + baserunningWAR + replacementWAR + positionalWAR
      battingWAR     = (wRC+ - 100) / 100 * playRatio * 5
      baserunningWAR = (盗塁 / 50) * 0.6
      replacementWAR = 2.0 * playRatio
      positionalWAR  = 守備位置補正 * playRatio
      playRatio      = min(1, 試合 / scheduled_games)
    """
    df = df.copy()
    play_ratio = (df["試合"].astype(float) / scheduled_games).clip(upper=1)
    df["wRC+"] = wrc_plus(df["OPS"]).round(0)

    batting_war = (df["wRC+"] - 100) / 100 * play_ratio * 5
    baserunning_war = df["盗塁"].astype(float) / 50 * 0.6
    replacement_war = 2.0 * play_ratio
    positional_war = df["position"].map(POSITION_WAR_ADJUSTMENT).fillna(0.0) * play_ratio

    df["WAR"] = (batting_war + baserunning_war + replacement_war + positional_war).round(2)
    return df


def league_fip_constant(team_pitching: pd.DataFrame) -> float:
    """そのリーグ・年度のFIP定数 (cFIP) を算出する。

    cFIP = 平均防御率 - (13*HR + 3*(BB+HBP) - 2*K) / IP
    """
    ip = team_pitching["投球回"].map(innings_to_float).sum()
    hr = team_pitching["本塁打"].astype(float).sum()
    bb = team_pitching["四球"].astype(float).sum()
    hbp = team_pitching["死球"].astype(float).sum()
    so = team_pitching["三振"].astype(float).sum()
    er = team_pitching["自責点"].astype(float).sum()
    league_era = er * 9 / ip
    raw = (13 * hr + 3 * (bb + hbp) - 2 * so) / ip
    return float(league_era - raw)


def add_pitching_sabermetrics(df: pd.DataFrame, fip_constant: float) -> pd.DataFrame:
    df = df.copy()
    ip = df["投球回"].map(innings_to_float)
    batters = df["打者"].astype(float)
    h = df["安打"].astype(float)
    hr = df["本塁打"].astype(float)
    bb = df["四球"].astype(float)
    hbp = df["死球"].astype(float)
    so = df["三振"].astype(float)

    ip_safe = ip.replace(0, np.nan)

    df["投球回_数値"] = ip.round(3)
    df["WHIP"] = ((h + bb) / ip_safe).round(2)
    df["K/9"] = (so * 9 / ip_safe).round(2)
    df["BB/9"] = (bb * 9 / ip_safe).round(2)
    df["K/BB"] = (so / bb.replace(0, np.nan)).round(2)
    df["K%"] = (so / batters.replace(0, np.nan) * 100).round(1)
    df["BB%"] = (bb / batters.replace(0, np.nan) * 100).round(1)
    df["FIP"] = ((13 * hr + 3 * (bb + hbp) - 2 * so) / ip_safe + fip_constant).round(2)

    return df


def pythagorean_win_pct(runs_scored, runs_allowed, exponent: float = 1.83):
    """ピタゴラス勝率。指数1.83はBaseball Prospectus等で使われる補正値 (Bill James原案は2)。"""
    rs = np.asarray(runs_scored, dtype=float)
    ra = np.asarray(runs_allowed, dtype=float)
    return rs**exponent / (rs**exponent + ra**exponent)


def add_team_sabermetrics(standings: pd.DataFrame, team_batting: pd.DataFrame, team_pitching: pd.DataFrame) -> pd.DataFrame:
    """順位表にピタゴラス勝率・得失点差・「運」(実勝率-期待勝率)を付加する。"""
    rs = team_batting.set_index("チーム")["得点"]
    ra = team_pitching.set_index("チーム")["失点"]

    df = standings.copy()
    df["得点"] = df["チーム"].map(rs)
    df["失点"] = df["チーム"].map(ra)
    df["得失点差"] = df["得点"] - df["失点"]
    df["ピタゴラス勝率"] = pythagorean_win_pct(df["得点"], df["失点"]).round(3)
    df["期待勝利数"] = (df["ピタゴラス勝率"] * df["試合"]).round(1)
    df["運(実勝利-期待勝利)"] = (df["勝利"] - df["期待勝利数"]).round(1)
    return df
