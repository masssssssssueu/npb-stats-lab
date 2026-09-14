"""順位予想・勝敗予想・選手成績予想。

シンプルだが実績のある古典的サイバーメトリクス手法を採用:
- 順位予想: ピタゴラス勝率でチームの地力を推定し、残り試合をモンテカルロ・シミュレーション
- 勝敗予想: log5法 (Bill James考案の対戦勝率推定式)
- 選手成績予想: スタッツの「安定化点」(Russell Carletonの研究に基づくサンプルサイズ)で
  リーグ平均に回帰させたうえで、残り試合分を現在の出場ペースで延伸する

本格的な機械学習モデルではなく、過去データが少なくても機能する統計的推定であることに留意。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sabermetrics import innings_to_float, pythagorean_win_pct

SCHEDULED_GAMES = 143  # NPBレギュラーシーズンの規定試合数

# Russell Carletonらの研究に基づく「安定化点」(このPA/IP数を超えると実力を反映し始める目安)
STABILIZATION_PA = {"打率": 910, "出塁率": 460, "長打率": 320, "ISO": 160, "BABIP": 820, "K%": 60, "BB%": 120}
STABILIZATION_IP_PITCHING = {"K%": 70, "BB%": 170, "防御率": 630, "WHIP": 240}


def recent_form(team: str, game_log: pd.DataFrame, n: int = 5) -> dict:
    """試合日程ログから、そのチームの直近n試合の成績を集計する。"""
    completed = game_log[game_log["completed"]]
    mask = (completed["team1"] == team) | (completed["team2"] == team)
    games = completed[mask].tail(n)

    results = []
    runs_for = runs_against = 0
    for _, g in games.iterrows():
        if g["team1"] == team:
            rf, ra = g["score1"], g["score2"]
        else:
            rf, ra = g["score2"], g["score1"]
        runs_for += rf
        runs_against += ra
        results.append("W" if rf > ra else ("L" if rf < ra else "D"))

    wins, losses, draws = results.count("W"), results.count("L"), results.count("D")
    decided = wins + losses
    return {
        "results": results,
        "record": f"{wins}勝{losses}敗{draws}分",
        "win_pct": wins / decided if decided else 0.5,
        "runs_for": runs_for,
        "runs_against": runs_against,
        "n_games": len(games),
    }


def blended_win_pct(season_pct: float, recent_pct: float, recent_weight: float = 0.35) -> float:
    """シーズン全体の勝率と直近成績の勝率を加重平均する(直近の調子を軽く加味する)。"""
    return (1 - recent_weight) * season_pct + recent_weight * recent_pct


def starter_fip_adjustment(home_fip: float, away_fip: float, scale: float = 0.05, cap: float = 0.15) -> float:
    """先発投手のFIP差を勝率への補正幅に変換する(FIPが1点低いほど有利、との簡易換算)。"""
    adj = (away_fip - home_fip) * scale
    return max(-cap, min(cap, adj))


def log5_win_prob(win_pct_a: float, win_pct_b: float) -> float:
    """log5法: 2チームの勝率からその対戦カードでのAチーム勝率を推定する。"""
    a, b = float(win_pct_a), float(win_pct_b)
    denom = a + b - 2 * a * b
    if denom <= 0:
        return 0.5
    return (a - a * b) / denom


def simulate_final_standings(
    team_df: pd.DataFrame,
    *,
    n_sim: int = 20000,
    scheduled_games: int = SCHEDULED_GAMES,
    exponent: float = 1.83,
    seed: int | None = 42,
) -> pd.DataFrame:
    """ピタゴラス勝率を「勝つ力」として残り試合をモンテカルロ・シミュレーションし、
    最終順位・CS進出(上位3位)確率・優勝確率を算出する。リーグ単位で呼び出すこと。
    """
    rng = np.random.default_rng(seed)
    teams = team_df["チーム"].tolist()
    n_teams = len(teams)

    wins = team_df["勝利"].to_numpy(dtype=float)
    played = team_df["試合"].to_numpy(dtype=float)
    remaining = np.clip(scheduled_games - played, 0, None)
    win_pct = pythagorean_win_pct(team_df["得点"], team_df["失点"], exponent=exponent)

    # (n_sim, n_teams) の残り試合勝利数をまとめて二項乱数で生成
    sim_extra_wins = rng.binomial(remaining.astype(int), win_pct, size=(n_sim, n_teams))
    sim_final_wins = wins[None, :] + sim_extra_wins
    sim_final_pct = sim_final_wins / scheduled_games

    # 順位 (勝率が高いほど上位、同率は乱数タイブレークで安定させる)
    tie_break = rng.random((n_sim, n_teams)) * 1e-9
    ranks = (-(sim_final_pct + tie_break)).argsort(axis=1).argsort(axis=1) + 1

    result = pd.DataFrame({
        "チーム": teams,
        "試合": played.astype(int),
        "勝利": wins.astype(int),
        "残り試合": remaining.astype(int),
        "ピタゴラス勝率": win_pct.round(3),
        "予想最終勝利数(平均)": sim_final_wins.mean(axis=0).round(1),
        "優勝確率": (ranks == 1).mean(axis=0).round(3),
        "CS進出確率(上位3位)": (ranks <= 3).mean(axis=0).round(3),
    })
    return result.sort_values("優勝確率", ascending=False).reset_index(drop=True)


def stabilize_rate(observed_rate: float, sample_size: float, stabilization_point: float, league_avg_rate: float) -> float:
    """観測値をサンプルサイズに応じてリーグ平均へ回帰させる (安定化点分のリーグ平均データを仮想的に加える)。"""
    if sample_size + stabilization_point <= 0:
        return league_avg_rate
    return (sample_size * observed_rate + stabilization_point * league_avg_rate) / (sample_size + stabilization_point)


def league_batting_averages(team_batting: pd.DataFrame) -> dict:
    ab = team_batting["打数"].astype(float).sum()
    h = team_batting["安打"].astype(float).sum()
    bb = team_batting["四球"].astype(float).sum()
    hbp = team_batting["死球"].astype(float).sum()
    sf = team_batting["犠飛"].astype(float).sum()
    tb = team_batting["塁打"].astype(float).sum()
    pa = team_batting["打席"].astype(float).sum()
    avg = h / ab
    obp = (h + bb + hbp) / (ab + bb + hbp + sf)
    slg = tb / ab
    return {"打率": avg, "出塁率": obp, "長打率": slg, "ISO": slg - avg, "BABIP": (h - team_batting["本塁打"].astype(float).sum()) / (ab - team_batting["三振"].astype(float).sum() - team_batting["本塁打"].astype(float).sum() + sf), "K%": team_batting["三振"].astype(float).sum() / pa * 100, "BB%": bb / pa * 100}


def project_batter_rest_of_season(
    batting_df: pd.DataFrame,
    team_batting: pd.DataFrame,
    standings: pd.DataFrame,
    *,
    scheduled_games: int = SCHEDULED_GAMES,
) -> pd.DataFrame:
    """安定化点でリーグ平均に回帰させた打率等をもとに、現在の出場ペースで残り試合分を延伸し
    最終着地成績 (打率・出塁率・長打率・OPS・予想本塁打数) を予想する。
    """
    league_avg = league_batting_averages(team_batting)
    games_played = standings.set_index("チーム")["試合"]
    remaining_games = (scheduled_games - games_played).clip(lower=0)

    df = batting_df.copy()
    team_games = df["球団"].map(games_played)
    team_remaining = df["球団"].map(remaining_games)
    pa_per_game = df["打席"] / team_games.replace(0, np.nan)
    proj_remaining_pa = (pa_per_game * team_remaining).fillna(0)
    current_pa = df["打席"]
    final_pa = (current_pa + proj_remaining_pa).replace(0, np.nan)

    for stat in ("打率", "出塁率", "長打率"):
        # 安定化点でリーグ平均へ回帰させた値は、あくまで「今後の実力推定」として残り試合の予想にのみ使う。
        # 既に確定している現在の実績まで平均へ引き戻すのは誤りなので、最終着地成績は
        # 「現在の実績(打席数分の重み)」+「残り試合の実力推定(残り打席数分の重み)」の加重平均にする。
        talent_rate = df.apply(
            lambda r, s=stat: stabilize_rate(r[s], r["打席"], STABILIZATION_PA[s], league_avg[s]), axis=1
        )
        df[f"回帰後{stat}"] = talent_rate.round(3)
        df[f"予想最終{stat}"] = (
            (df[stat] * current_pa + talent_rate * proj_remaining_pa) / final_pa
        ).round(3)

    df["予想最終OPS"] = (df["予想最終出塁率"] + df["予想最終長打率"]).round(3)
    hr_rate = df["本塁打"] / df["打席"].replace(0, np.nan)
    df["予想最終本塁打数"] = (df["本塁打"] + hr_rate * proj_remaining_pa).round(1)

    return df


def league_pitching_averages(team_pitching: pd.DataFrame) -> dict:
    ip = team_pitching["投球回"].map(innings_to_float).sum()
    h = team_pitching["安打"].astype(float).sum()
    bb = team_pitching["四球"].astype(float).sum()
    so = team_pitching["三振"].astype(float).sum()
    er = team_pitching["自責点"].astype(float).sum()
    batters = team_pitching["打者"].astype(float).sum()
    return {
        "防御率": er * 9 / ip,
        "WHIP": (h + bb) / ip,
        "K%": so / batters * 100,
        "BB%": bb / batters * 100,
    }


def project_pitcher_rest_of_season(
    pitching_df: pd.DataFrame,
    team_pitching: pd.DataFrame,
    standings: pd.DataFrame,
    *,
    scheduled_games: int = SCHEDULED_GAMES,
) -> pd.DataFrame:
    """安定化点でリーグ平均に回帰させたERA・WHIP・K%等をもとに、現在の登板ペースで
    残り試合分を延伸し、シーズン最終着地の投球回・奪三振数を予想する。
    """
    league_avg = league_pitching_averages(team_pitching)
    games_played = standings.set_index("チーム")["試合"]
    remaining_games = (scheduled_games - games_played).clip(lower=0)

    df = pitching_df.copy()
    df["_ip"] = df["投球回"].map(innings_to_float)
    team_games = df["球団"].map(games_played)
    team_remaining = df["球団"].map(remaining_games)
    ip_per_game = df["_ip"] / team_games.replace(0, np.nan)
    proj_remaining_ip = (ip_per_game * team_remaining).fillna(0)
    batters_per_game = df["打者"] / team_games.replace(0, np.nan)
    proj_remaining_batters = (batters_per_game * team_remaining).fillna(0)

    final_ip = (df["_ip"] + proj_remaining_ip).replace(0, np.nan)

    talent_era = df.apply(
        lambda r: stabilize_rate(r["防御率"], r["_ip"], STABILIZATION_IP_PITCHING["防御率"], league_avg["防御率"]), axis=1
    )
    talent_whip = df.apply(
        lambda r: stabilize_rate(r["WHIP"], r["_ip"], STABILIZATION_IP_PITCHING["WHIP"], league_avg["WHIP"]), axis=1
    )
    regressed_k_pct = df.apply(
        lambda r: stabilize_rate(r["K%"], r["打者"], STABILIZATION_IP_PITCHING["K%"], league_avg["K%"]), axis=1
    )
    df["回帰後防御率"] = talent_era.round(2)
    df["回帰後WHIP"] = talent_whip.round(2)

    # 既に投げた分の実績まで平均へ引き戻さないよう、最終着地成績は
    # 「現在の実績(投球回分の重み)」+「残り登板の実力推定(残り投球回分の重み)」の加重平均にする。
    df["予想最終防御率"] = ((df["防御率"] * df["_ip"] + talent_era * proj_remaining_ip) / final_ip).round(2)
    df["予想最終WHIP"] = ((df["WHIP"] * df["_ip"] + talent_whip * proj_remaining_ip) / final_ip).round(2)

    df["予想最終投球回"] = (df["_ip"] + proj_remaining_ip).round(1)
    df["予想最終奪三振数"] = (df["三振"] + regressed_k_pct / 100 * proj_remaining_batters).round(1)

    decisions = df["勝利"].astype(float) + df["敗北"].astype(float)
    decision_rate = decisions / df["_ip"].replace(0, np.nan)
    remaining_decisions = (decision_rate * proj_remaining_ip).fillna(0)
    win_share_in_decisions = (df["勝利"].astype(float) / decisions.replace(0, np.nan)).fillna(0.5)
    df["予想最終勝利数"] = (df["勝利"] + remaining_decisions * win_share_in_decisions).round(1)
    df["予想最終敗戦数"] = (df["敗北"] + remaining_decisions * (1 - win_share_in_decisions)).round(1)

    return df.drop(columns="_ip")
