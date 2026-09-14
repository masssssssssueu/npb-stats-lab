"""球団名の表記ゆれ(公式フルネーム/短縮名/一文字略称)を正規化する。"""

# 正規化後の短縮名 (チーム統計ページ tmb_*/tmp_* の表記に合わせる)
CENTRAL_TEAMS = ["阪神", "巨人", "DeNA", "広島", "中日", "ヤクルト"]
PACIFIC_TEAMS = ["ソフトバンク", "西武", "日本ハム", "オリックス", "ロッテ", "楽天"]
ALL_TEAMS = CENTRAL_TEAMS + PACIFIC_TEAMS

# 選手個人成績ページの <span class="stteam">(神)</span> 等、一文字略称 -> 短縮名
ABBR_TO_SHORT = {
    "神": "阪神", "巨": "巨人", "デ": "DeNA", "広": "広島", "中": "中日", "ヤ": "ヤクルト",
    "ソ": "ソフトバンク", "西": "西武", "日": "日本ハム", "オ": "オリックス", "ロ": "ロッテ", "楽": "楽天",
}

# 順位表ページ (std_c/std_p) の正式球団名 -> 短縮名
FULLNAME_TO_SHORT = {
    "阪神タイガース": "阪神",
    "読売ジャイアンツ": "巨人",
    "横浜DeNAベイスターズ": "DeNA",
    "広島東洋カープ": "広島",
    "中日ドラゴンズ": "中日",
    "東京ヤクルトスワローズ": "ヤクルト",
    "福岡ソフトバンクホークス": "ソフトバンク",
    "埼玉西武ライオンズ": "西武",
    "北海道日本ハムファイターズ": "日本ハム",
    "オリックス・バファローズ": "オリックス",
    "千葉ロッテマリーンズ": "ロッテ",
    "東北楽天ゴールデンイーグルス": "楽天",
}

TEAM_COLORS = {
    "阪神": "#F5C623", "巨人": "#F97709", "DeNA": "#0057A6", "広島": "#E5001C",
    "中日": "#00447A", "ヤクルト": "#8FC31F",
    "ソフトバンク": "#FFD500", "西武": "#00509A", "日本ハム": "#5EBCE4",
    "オリックス": "#001E62", "ロッテ": "#000000", "楽天": "#870139",
}


def normalize_team(name: str) -> str:
    name = name.strip()
    if name in ALL_TEAMS:
        return name
    if name in ABBR_TO_SHORT:
        return ABBR_TO_SHORT[name]
    if name in FULLNAME_TO_SHORT:
        return FULLNAME_TO_SHORT[name]
    raise KeyError(f"未知の球団表記です: {name!r}")


def league_of(team_short: str) -> str:
    if team_short in CENTRAL_TEAMS:
        return "セ"
    if team_short in PACIFIC_TEAMS:
        return "パ"
    raise KeyError(f"未知の球団です: {team_short!r}")
