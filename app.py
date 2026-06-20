import time
import re
from datetime import datetime, timezone, timedelta
from urllib.parse import unquote, urljoin, quote
from io import BytesIO
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"


PRESETS = {
    25: {
        "厳しめ": {
            # 出来高急増起点からの初押しをかなり厳しく探す
            "spike_lookback": 30,
            "volume_spike_mult": 5.0,
            "min_days_since_spike": 5,
            "rise_window": 8,
            "min_rise_pct": 10.0,
            "near_lower": -1.5,
            "near_upper": 3.0,
            "min_pullback_pct": 4.0,
            "first_touch_upper": 3.5,
            "ma25_slope_lookback": 10,
            "ma25_slope_min": 0.3,
            "max_loss_pct": 7.0,
        },
        "普通": {
            # 最初に使うおすすめ設定
            "spike_lookback": 40,
            "volume_spike_mult": 5.0,
            "min_days_since_spike": 4,
            "rise_window": 10,
            "min_rise_pct": 7.0,
            "near_lower": -2.5,
            "near_upper": 4.5,
            "min_pullback_pct": 3.0,
            "first_touch_upper": 5.0,
            "ma25_slope_lookback": 10,
            "ma25_slope_min": 0.0,
            "max_loss_pct": 9.0,
        },
        "ゆるめ": {
            # 候補が少ないとき用。出来高5倍条件は維持する
            "spike_lookback": 50,
            "volume_spike_mult": 5.0,
            "min_days_since_spike": 3,
            "rise_window": 12,
            "min_rise_pct": 5.0,
            "near_lower": -3.5,
            "near_upper": 6.0,
            "min_pullback_pct": 2.0,
            "first_touch_upper": 6.5,
            "ma25_slope_lookback": 10,
            "ma25_slope_min": -0.5,
            "max_loss_pct": 11.0,
        },
    },
    75: {
        "超厳しめ": {
            "near_lower": -1.0,
            "near_upper": 2.0,
            "slope_lookback": 20,
            "slope_min": 1.5,
            "along_days": 5,
            "min_near_days": 3,
            "volume_ratio_max": 1.2,
        },
        "厳しめ": {
            "near_lower": -1.5,
            "near_upper": 3.0,
            "slope_lookback": 20,
            "slope_min": 1.0,
            "along_days": 5,
            "min_near_days": 3,
            "volume_ratio_max": 1.4,
        },
        "普通": {
            "near_lower": -2.5,
            "near_upper": 5.0,
            "slope_lookback": 20,
            "slope_min": 0.5,
            "along_days": 5,
            "min_near_days": 2,
            "volume_ratio_max": 1.8,
        },
        "ゆるめ": {
            "near_lower": -4.0,
            "near_upper": 8.0,
            "slope_lookback": 20,
            "slope_min": 0.0,
            "along_days": 5,
            "min_near_days": 1,
            "volume_ratio_max": 2.5,
        },
    },
    200: {
        "超厳しめ": {
            "near_lower": -1.5,
            "near_upper": 3.0,
            "slope_lookback": 45,
            "slope_min": 3.0,
            "along_days": 8,
            "min_near_days": 4,
            "volume_ratio_max": 1.2,
        },
        "厳しめ": {
            "near_lower": -2.0,
            "near_upper": 4.5,
            "slope_lookback": 45,
            "slope_min": 2.0,
            "along_days": 8,
            "min_near_days": 3,
            "volume_ratio_max": 1.5,
        },
        "普通": {
            "near_lower": -3.5,
            "near_upper": 6.5,
            "slope_lookback": 45,
            "slope_min": 0.8,
            "along_days": 8,
            "min_near_days": 2,
            "volume_ratio_max": 2.0,
        },
        "ゆるめ": {
            "near_lower": -5.0,
            "near_upper": 10.0,
            "slope_lookback": 45,
            "slope_min": -0.5,
            "along_days": 8,
            "min_near_days": 1,
            "volume_ratio_max": 2.8,
        },
    },
}


# 三角保ち合いブレイク用プリセット
# 前回版でこの定義が抜けていたため、Streamlit側で NameError が出ていました。
TRIANGLE_PRESETS = {
    "厳しめ": {
        "lookback": 40,
        "breakout_volume_min": 2.2,
        "range_ratio_max": 0.70,
        "volume_contract_ratio_max": 0.75,
        "high_slope_max_pct_per_day": 0.02,
        "low_slope_min_pct_per_day": 0.02,
        "close_position_min": 0.60,
        "stop_loss_max_pct": 7.0,
        "rr_min": 1.5,
        "need_ma75_up": True,
    },
    "普通": {
        "lookback": 40,
        "breakout_volume_min": 2.0,
        "range_ratio_max": 0.80,
        "volume_contract_ratio_max": 0.90,
        "high_slope_max_pct_per_day": 0.05,
        "low_slope_min_pct_per_day": -0.01,
        "close_position_min": 0.50,
        "stop_loss_max_pct": 10.0,
        "rr_min": 1.2,
        "need_ma75_up": False,
    },
    "ゆるめ": {
        "lookback": 30,
        "breakout_volume_min": 1.5,
        "range_ratio_max": 0.90,
        "volume_contract_ratio_max": 1.10,
        "high_slope_max_pct_per_day": 0.10,
        "low_slope_min_pct_per_day": -0.05,
        "close_position_min": 0.45,
        "stop_loss_max_pct": 12.0,
        "rr_min": 1.0,
        "need_ma75_up": False,
    },
}



# ウェッジブレイク（WB）/ ウェッジブレイク・プルバック（WBPB）用プリセット
# 上昇トレンドの途中で形成される「下降ウェッジ」を対象にします。
# WB：下降トレンドラインを出来高を伴って上抜ける初動
# WBPB：初回ブレイク後に押しを作り、直近スイング高値を再度上抜ける形
WEDGE_PRESETS = {
    "厳しめ": {
        "lookback": 22,
        "prior_trend_days": 20,
        "min_prior_rise_pct": 10.0,
        "upper_slope_max_pct_per_day": -0.015,
        "convergence_ratio_max": 0.70,
        "min_upper_touches": 2,
        "min_lower_touches": 2,
        "touch_tolerance_pct": 1.8,
        "volume_contract_ratio_max": 0.90,
        "breakout_volume_min": 1.8,
        "pullback_volume_min": 1.2,
        "close_position_min": 0.60,
        "monitor_distance_pct": 3.0,
        "breakout_margin_pct": 0.20,
        "pullback_min_pct": 2.0,
        "pullback_max_days": 10,
        "stop_loss_max_pct": 7.0,
        "rr_min": 1.3,
        "need_ma75_up": True,
    },
    "普通": {
        "lookback": 20,
        "prior_trend_days": 15,
        "min_prior_rise_pct": 6.0,
        "upper_slope_max_pct_per_day": -0.005,
        "convergence_ratio_max": 0.82,
        "min_upper_touches": 2,
        "min_lower_touches": 2,
        "touch_tolerance_pct": 2.3,
        "volume_contract_ratio_max": 1.10,
        "breakout_volume_min": 1.3,
        "pullback_volume_min": 1.0,
        "close_position_min": 0.50,
        "monitor_distance_pct": 4.0,
        "breakout_margin_pct": 0.00,
        "pullback_min_pct": 1.5,
        "pullback_max_days": 12,
        "stop_loss_max_pct": 10.0,
        "rr_min": 1.0,
        "need_ma75_up": False,
    },
    "ゆるめ": {
        "lookback": 16,
        "prior_trend_days": 12,
        "min_prior_rise_pct": 3.0,
        "upper_slope_max_pct_per_day": 0.010,
        "convergence_ratio_max": 0.95,
        "min_upper_touches": 1,
        "min_lower_touches": 1,
        "touch_tolerance_pct": 3.0,
        "volume_contract_ratio_max": 1.25,
        "breakout_volume_min": 1.0,
        "pullback_volume_min": 0.9,
        "close_position_min": 0.45,
        "monitor_distance_pct": 5.0,
        "breakout_margin_pct": 0.00,
        "pullback_min_pct": 1.0,
        "pullback_max_days": 15,
        "stop_loss_max_pct": 12.0,
        "rr_min": 0.8,
        "need_ma75_up": False,
    },
}



# 週足ウェッジ用プリセット。
# 日足よりも保ち合い期間・出来高の単位が大きくなるため、
# 日足版とは別条件でWB/WBPBを判定する。
WEDGE_WEEKLY_PRESETS = {
    "厳しめ": {
        "lookback": 18,
        "prior_trend_days": 20,
        "min_prior_rise_pct": 15.0,
        "upper_slope_max_pct_per_day": -0.02,
        "convergence_ratio_max": 0.72,
        "min_upper_touches": 2,
        "min_lower_touches": 2,
        "touch_tolerance_pct": 2.0,
        "volume_contract_ratio_max": 0.95,
        "breakout_volume_min": 1.5,
        "pullback_volume_min": 1.0,
        "close_position_min": 0.60,
        "monitor_distance_pct": 3.0,
        "breakout_margin_pct": 0.15,
        "pullback_min_pct": 3.0,
        "pullback_max_days": 8,
        "stop_loss_max_pct": 10.0,
        "rr_min": 1.2,
        "need_ma75_up": True,
    },
    "普通": {
        "lookback": 16,
        "prior_trend_days": 16,
        "min_prior_rise_pct": 10.0,
        "upper_slope_max_pct_per_day": 0.00,
        "convergence_ratio_max": 0.85,
        "min_upper_touches": 2,
        "min_lower_touches": 2,
        "touch_tolerance_pct": 2.5,
        "volume_contract_ratio_max": 1.15,
        "breakout_volume_min": 1.2,
        "pullback_volume_min": 0.9,
        "close_position_min": 0.50,
        "monitor_distance_pct": 4.0,
        "breakout_margin_pct": 0.00,
        "pullback_min_pct": 2.0,
        "pullback_max_days": 10,
        "stop_loss_max_pct": 13.0,
        "rr_min": 0.9,
        "need_ma75_up": False,
    },
    "ゆるめ": {
        "lookback": 14,
        "prior_trend_days": 14,
        "min_prior_rise_pct": 6.0,
        "upper_slope_max_pct_per_day": 0.05,
        "convergence_ratio_max": 0.98,
        "min_upper_touches": 1,
        "min_lower_touches": 1,
        "touch_tolerance_pct": 3.2,
        "volume_contract_ratio_max": 1.35,
        "breakout_volume_min": 1.0,
        "pullback_volume_min": 0.8,
        "close_position_min": 0.45,
        "monitor_distance_pct": 5.0,
        "breakout_margin_pct": 0.00,
        "pullback_min_pct": 1.5,
        "pullback_max_days": 12,
        "stop_loss_max_pct": 16.0,
        "rr_min": 0.7,
        "need_ma75_up": False,
    },
}


# モメンタム初動・強勢継続用プリセット
# 「強いものを強いまま買う」相場用。
# 高値更新・出来高増・短期上昇・移動平均線上の強さをスコア化します。
MOMENTUM_PRESETS = {
    "超厳しめ": {
        # 候補をかなり絞る設定。
        # 「本当に資金が入っている高値更新初動」だけを優先します。
        "min_score_candidate": 95,
        "min_score_watch": 85,
        "min_ret_3d": 4.0,
        "min_ret_5d": 15.0,
        "min_ret_20d": 25.0,
        "high20_distance_max": 0.3,
        "high60_distance_max": 2.0,
        "volume_ratio_min": 3.0,
        "volume_ratio20_min": 2.0,
        "close_position_min": 0.80,
        "ma25_deviation_max": 12.0,
        "stop_loss_max_pct": 5.0,
        "upper_wick_max_pct": 15.0,
        "require_bullish": True,
        "require_ma75_up": True,
        "hard_candidate_filter": True,
        "max_watch_failures": 1,
    },
    "厳しめ": {
        # 以前よりかなり厳しくしています。
        # 候補が多すぎる場合は、まずこの設定を使ってください。
        "min_score_candidate": 90,
        "min_score_watch": 78,
        "min_ret_3d": 2.5,
        "min_ret_5d": 12.0,
        "min_ret_20d": 18.0,
        "high20_distance_max": 0.8,
        "high60_distance_max": 4.0,
        "volume_ratio_min": 2.5,
        "volume_ratio20_min": 1.5,
        "close_position_min": 0.75,
        "ma25_deviation_max": 15.0,
        "stop_loss_max_pct": 6.0,
        "upper_wick_max_pct": 22.0,
        "require_bullish": True,
        "require_ma75_up": True,
        "hard_candidate_filter": True,
        "max_watch_failures": 2,
    },
    "普通": {
        # 普通も少しだけ厳しめに調整。
        "min_score_candidate": 78,
        "min_score_watch": 62,
        "min_ret_3d": 0.0,
        "min_ret_5d": 9.0,
        "min_ret_20d": 12.0,
        "high20_distance_max": 2.0,
        "high60_distance_max": 8.0,
        "volume_ratio_min": 1.8,
        "volume_ratio20_min": 1.1,
        "close_position_min": 0.65,
        "ma25_deviation_max": 22.0,
        "stop_loss_max_pct": 8.0,
        "upper_wick_max_pct": 35.0,
        "require_bullish": False,
        "require_ma75_up": True,
        "hard_candidate_filter": True,
        "max_watch_failures": 3,
    },
    "ゆるめ": {
        "min_score_candidate": 65,
        "min_score_watch": 50,
        "min_ret_3d": 0.0,
        "min_ret_5d": 6.0,
        "min_ret_20d": 8.0,
        "high20_distance_max": 4.0,
        "high60_distance_max": 12.0,
        "volume_ratio_min": 1.3,
        "volume_ratio20_min": 1.0,
        "close_position_min": 0.55,
        "ma25_deviation_max": 30.0,
        "stop_loss_max_pct": 10.0,
        "upper_wick_max_pct": 45.0,
        "require_bullish": False,
        "require_ma75_up": False,
        "hard_candidate_filter": False,
        "max_watch_failures": 4,
    },
}

# テーマ判定用。
# 完全な分類ではなく、銘柄名・業種・一部コードから「何の物色か」を見やすくする補助表示です。
THEME_CODE_OVERRIDES = {
    "4004": "HDD/半導体材料/化学",
    "6762": "HDD/電子部品/MLCC周辺",
    "6594": "HDD/モーター/精密部品",
    "6479": "HDD/精密部品/ベアリング",
    "402A": "宇宙/小型衛星/防衛",
    "5595": "宇宙/衛星データ",
    "7011": "防衛/重工/原発",
    "7012": "防衛/造船/重工",
    "7013": "防衛/航空機/重工",
    "6208": "防衛/機械",
    "5801": "電線/電力インフラ/非鉄",
    "5802": "電線/電力インフラ/非鉄",
    "5803": "電線/電力インフラ/非鉄",
    "6976": "MLCC/電子部品",
    "6981": "電子部品/スマホ/車載",
    "6723": "半導体製造装置",
    "6857": "半導体製造装置",
    "6920": "半導体検査装置",
    "8035": "半導体製造装置",
    "7735": "半導体製造装置",
    "6315": "半導体製造装置/真空",
    "6701": "AI/IT/防衛",
    "6702": "AI/IT/通信",
    "6501": "電力インフラ/原発/AIデータセンター",
    "6503": "電力インフラ/FA/防衛",
    "6504": "電力インフラ/重電",
    "9501": "電力/原発/インフラ",
    "9502": "電力/原発/インフラ",
    "9503": "電力/原発/インフラ",
}

THEME_KEYWORDS = [
    ("宇宙", ["宇宙", "衛星", "スペース", "ispace", "アイスペース", "アクセルスペース"]),
    ("防衛", ["防衛", "重工", "造船", "航空", "火工", "日本アビオ", "細谷火工", "豊和工"]),
    ("半導体", ["半導体", "東京エレクトロン", "レーザーテック", "ディスコ", "アドバンテスト", "SCREEN", "ＳＣＲＥＥＮ", "芝浦メカ", "ローツェ"]),
    ("AI/データセンター", ["AI", "ＡＩ", "データセンター", "サーバー", "電算", "情報", "システム", "ソフト", "クラウド"]),
    ("電線/電力インフラ", ["電線", "古河電", "住友電", "フジクラ", "電設", "変圧", "電力", "電工"]),
    ("MLCC/電子部品", ["村田", "ＴＤＫ", "太陽誘電", "電子部品", "コンデンサ", "セラミック"]),
    ("HDD/ストレージ", ["HDD", "ＨＤＤ", "ハードディスク", "レゾナック", "ニデック", "ミネベア"]),
    ("原発/電力", ["原子力", "原発", "電力", "東京電力", "関西電力", "中部電力", "九州電力", "北海道電力"]),
    ("造船/海運", ["造船", "海運", "郵船", "商船", "川崎汽船"]),
    ("バイオ/医薬", ["ファーマ", "製薬", "バイオ", "創薬", "メディシノバ", "ペプチド", "そーせい"]),
    ("インバウンド/小売", ["百貨店", "免税", "ホテル", "旅行", "リゾート", "ワークマン", "トライアル", "パンパシ"]),
    ("金融/暗号資産", ["銀行", "証券", "保険", "ビットコイン", "暗号資産", "仮想通貨", "ホドル"]),
    ("資源/レアアース", ["鉱業", "石油", "資源", "レアアース", "非鉄", "金属", "商事", "物産"]),
]


# カップウィズハンドル用プリセット
# 日足・週足・月足それぞれで、カップ形成後にピボット上抜けが近いものを探します。
CUP_PRESETS = {
    "厳しめ": {
        "pivot_distance_max_pct": 3.0,
        "breakout_over_allow_pct": 3.0,
        "rim_tolerance_pct": 7.0,
        "min_cup_depth_pct": 12.0,
        "max_cup_depth_pct": 45.0,
        "min_handle_depth_pct": 2.0,
        "max_handle_depth_pct": 14.0,
        "handle_vs_cup_max": 0.45,
        "need_volume_contract": True,
        "need_ma_trend": True,
    },
    "普通": {
        "pivot_distance_max_pct": 6.0,
        "breakout_over_allow_pct": 4.0,
        "rim_tolerance_pct": 10.0,
        "min_cup_depth_pct": 10.0,
        "max_cup_depth_pct": 55.0,
        "min_handle_depth_pct": 1.5,
        "max_handle_depth_pct": 18.0,
        "handle_vs_cup_max": 0.55,
        "need_volume_contract": False,
        "need_ma_trend": False,
    },
    "ゆるめ": {
        "pivot_distance_max_pct": 10.0,
        "breakout_over_allow_pct": 6.0,
        "rim_tolerance_pct": 15.0,
        "min_cup_depth_pct": 8.0,
        "max_cup_depth_pct": 65.0,
        "min_handle_depth_pct": 0.8,
        "max_handle_depth_pct": 25.0,
        "handle_vs_cup_max": 0.70,
        "need_volume_contract": False,
        "need_ma_trend": False,
    },
}

CUP_TIMEFRAME_SETTINGS = {
    "日足": {
        "resample_rule": None,
        "cup_min": 45,
        "cup_max": 180,
        "cup_step": 10,
        "handle_min": 5,
        "handle_max": 25,
        "handle_step": 2,
        "min_bars": 120,
    },
    "週足": {
        "resample_rule": "W-FRI",
        "cup_min": 20,
        "cup_max": 90,
        "cup_step": 4,
        "handle_min": 3,
        "handle_max": 14,
        "handle_step": 1,
        "min_bars": 70,
    },
    "月足": {
        "resample_rule": "M",
        "cup_min": 10,
        "cup_max": 48,
        "cup_step": 2,
        "handle_min": 2,
        "handle_max": 8,
        "handle_step": 1,
        "min_bars": 36,
    },
}

st.set_page_config(
    page_title="25日線初押し・75日線押し目・三角保ち合い・モメンタム",
    layout="wide"
)


def find_column(columns, keyword):
    for col in columns:
        if keyword in str(col):
            return col
    return None


def find_column_preferred(columns, keyword, preferred_words=None, exclude_words=None):
    """
    JPX一覧には「33業種コード」と「33業種区分」の両方がある。
    以前の版は先に見つかった「33業種コード」を使ってしまい、
    テーマ欄が 3650 などの数字になることがありました。
    この関数では「区分」「業種名」を優先し、「コード」は除外します。
    """
    preferred_words = preferred_words or []
    exclude_words = exclude_words or []
    cols = [str(c).strip() for c in columns]

    def ok(col):
        return keyword in col and not any(ex in col for ex in exclude_words)

    # まず完全一致・優先語を含む列を探す
    for pref in preferred_words:
        for col in cols:
            if ok(col) and pref in col:
                return col

    # 次に除外語を避けて探す
    for col in cols:
        if ok(col):
            return col

    # 最後の保険：従来の曖昧検索
    return find_column(cols, keyword)


def is_bad_theme_text(value):
    """
    テーマとして表示したくない値を判定する。
    例：33業種コードの 3650、銘柄コードの 402A など。
    """
    text = str(value).strip()
    if not text or text.lower() in ["nan", "none", "null"]:
        return True
    if text in ["-", "－", "—"]:
        return True
    text2 = text.replace(".0", "")
    if re.fullmatch(r"\d{1,6}", text2):
        return True
    # 4桁英数字で数字を含むものは銘柄コード扱いにする。AIやHDD等は除外されない。
    if re.fullmatch(r"(?=.*\d)[0-9A-Z]{4}", text2.upper()):
        return True
    return False


def clean_theme_source(value):
    text = str(value).strip()
    return "" if is_bad_theme_text(text) else text


def infer_theme(code, name, sector33="", sector17=""):
    """
    銘柄名・業種・一部コードからテーマを推定する。
    あくまで補助表示。材料確認は別途必要。
    """
    code = str(code).strip().upper().replace(".T", "")
    name_text = str(name)
    sector33 = clean_theme_source(sector33)
    sector17 = clean_theme_source(sector17)
    sector_text = f"{sector33} {sector17}"
    target = f"{code} {name_text} {sector_text}".upper()

    if code in THEME_CODE_OVERRIDES:
        return THEME_CODE_OVERRIDES[code]

    hits = []
    for theme, keywords in THEME_KEYWORDS:
        for kw in keywords:
            if str(kw).upper() in target:
                hits.append(theme)
                break

    # 業種ベースの補助分類
    sector_theme_map = {
        "電気機器": "電子部品/半導体/電機",
        "機械": "機械/半導体装置/FA",
        "精密機器": "精密/医療/半導体周辺",
        "情報・通信業": "IT/AI/クラウド",
        "医薬品": "医薬/バイオ",
        "非鉄金属": "非鉄/電線/資源",
        "鉱業": "資源/エネルギー",
        "石油・石炭製品": "資源/エネルギー",
        "電気・ガス業": "電力/インフラ",
        "海運業": "海運/市況",
        "空運業": "空運/インバウンド",
        "小売業": "小売/消費/インバウンド",
        "銀行業": "銀行/金利",
        "証券、商品先物取引業": "証券/金融",
        "保険業": "保険/金利",
        "不動産業": "不動産/金利",
        "建設業": "建設/インフラ",
        "化学": "化学/半導体材料",
        "サービス業": "サービス/人材/外食/娯楽",
    }

    if not hits:
        for sector_key, theme in sector_theme_map.items():
            if sector_key in sector_text:
                hits.append(theme)
                break

    if not hits:
        if sector33 and not is_bad_theme_text(sector33):
            hits.append(str(sector33))
        elif sector17 and not is_bad_theme_text(sector17):
            hits.append(str(sector17))
        else:
            hits.append("個別テーマ要確認")

    # 重複除去して最大3つまで
    unique_hits = []
    for h in hits:
        if h not in unique_hits:
            unique_hits.append(h)

    return " / ".join(unique_hits[:3])


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_jpx_list():
    response = requests.get(JPX_LIST_URL, timeout=30)
    response.raise_for_status()

    df = pd.read_excel(BytesIO(response.content))
    df.columns = [str(c).strip() for c in df.columns]

    code_col = find_column(df.columns, "コード")
    name_col = find_column(df.columns, "銘柄名")
    market_col = find_column(df.columns, "市場")

    if code_col is None or name_col is None or market_col is None:
        raise ValueError("JPXリストの列名を取得できませんでした。")

    # 「33業種コード」ではなく「33業種区分」を使う。ここがズレるとテーマが数字になります。
    sector33_col = find_column_preferred(
        df.columns,
        "33業種",
        preferred_words=["区分", "業種名", "分類"],
        exclude_words=["コード"],
    )
    sector17_col = find_column_preferred(
        df.columns,
        "17業種",
        preferred_words=["区分", "業種名", "分類"],
        exclude_words=["コード"],
    )

    use_cols = [code_col, name_col, market_col]
    if sector33_col is not None:
        use_cols.append(sector33_col)
    if sector17_col is not None and sector17_col != sector33_col:
        use_cols.append(sector17_col)

    result = df[use_cols].copy()

    rename_map = {
        code_col: "コード",
        name_col: "銘柄名",
        market_col: "市場",
    }
    if sector33_col is not None:
        rename_map[sector33_col] = "33業種"
    if sector17_col is not None and sector17_col != sector33_col:
        rename_map[sector17_col] = "17業種"

    result = result.rename(columns=rename_map)

    if "33業種" not in result.columns:
        result["33業種"] = ""
    if "17業種" not in result.columns:
        result["17業種"] = ""

    result["コード"] = (
        result["コード"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.upper()
    )

    result["銘柄名"] = result["銘柄名"].astype(str).str.strip()
    result["市場"] = result["市場"].astype(str).str.strip()
    result["33業種"] = result["33業種"].astype(str).str.strip().replace("nan", "")
    result["17業種"] = result["17業種"].astype(str).str.strip().replace("nan", "")
    result["33業種"] = result["33業種"].apply(clean_theme_source)
    result["17業種"] = result["17業種"].apply(clean_theme_source)

    result["テーマ"] = result.apply(
        lambda r: infer_theme(
            code=r.get("コード", ""),
            name=r.get("銘柄名", ""),
            sector33=r.get("33業種", ""),
            sector17=r.get("17業種", ""),
        ),
        axis=1,
    )

    result = result[result["コード"].str.match(r"^[0-9A-Z]{4}$", na=False)]
    return result.drop_duplicates(subset=["コード"]).reset_index(drop=True)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_price(code):
    ticker = f"{code}.T"

    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        first_level = df.columns.get_level_values(0)
        if "Close" in first_level:
            df.columns = first_level
        else:
            df.columns = df.columns.get_level_values(-1)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[required_cols].copy()
    df = df.dropna(subset=["Close"])

    try:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    return df



@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_price_long(code):
    """
    カップウィズハンドル用。
    月足も判定するため、通常より長めに5年分取得する。
    """
    ticker = f"{code}.T"

    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        first_level = df.columns.get_level_values(0)
        if "Close" in first_level:
            df.columns = first_level
        else:
            df.columns = df.columns.get_level_values(-1)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[required_cols].copy()
    df = df.dropna(subset=["Close"])

    try:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    return df

def filter_market(all_stocks, market_choice):
    """
    市場フィルター。
    全市場を選んだ場合は、プライム・スタンダード・グロースをまとめて対象にする。
    ETF、REIT、PRO Marketなどは除外。
    """
    if market_choice == "全市場":
        pattern = "プライム|スタンダード|グロース"
        return all_stocks[
            all_stocks["市場"].str.contains(pattern, na=False)
        ].copy()

    return all_stocks[
        all_stocks["市場"].str.contains(market_choice, na=False)
    ].copy()


def evaluate_extra_signals(
    df,
    ma_period,
    latest_close,
    latest_ma,
    latest_ma25,
    latest_ma75,
    latest_ma200,
    dist_ma_pct,
    slope_pct,
    volume_ratio,
    upside_to_20high_pct
):
    """
    優先度・反発確認・危険サイン・出来高評価を作る。
    売買判断の補助用なので、最終判断は必ずチャート確認する。
    """
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    latest_open = open_.iloc[-1]
    latest_high = high.iloc[-1]
    latest_volume = volume.iloc[-1]

    prev_close = close.iloc[-2] if len(close) >= 2 else np.nan
    prev_high = high.iloc[-2] if len(high) >= 2 else np.nan

    prev5_volume = volume.iloc[-6:-1].mean() if len(volume) >= 6 else np.nan
    vol20 = volume.tail(20).mean()

    is_bullish_candle = latest_close > latest_open
    is_close_up = pd.notna(prev_close) and latest_close > prev_close
    is_high_break_prev = pd.notna(prev_high) and latest_high > prev_high
    is_volume_rebound = pd.notna(prev5_volume) and latest_volume >= prev5_volume * 0.9

    rebound_score = sum([
        bool(is_bullish_candle),
        bool(is_close_up),
        bool(is_high_break_prev),
        bool(is_volume_rebound)
    ])

    if rebound_score >= 3:
        rebound_status = "あり"
    elif rebound_score == 2:
        rebound_status = "弱め"
    else:
        rebound_status = "待ち"

    recent_5_close = close.tail(5)
    recent_5_down_days = int((recent_5_close.diff() < 0).sum())

    high_volume_drop = (
        pd.notna(prev_close)
        and latest_close < prev_close
        and pd.notna(vol20)
        and vol20 > 0
        and latest_volume > vol20 * 1.3
    )

    danger_signs = []

    if dist_ma_pct < -1.5:
        danger_signs.append(f"{ma_period}日線をやや割れ")

    if pd.notna(latest_ma25) and pd.notna(latest_ma75):
        if latest_ma25 < latest_ma75:
            danger_signs.append("25日線<75日線")

    if pd.notna(latest_ma200) and latest_close < latest_ma200:
        danger_signs.append("200日線割れ")

    if recent_5_down_days >= 3:
        danger_signs.append("直近5日で下落多め")

    if high_volume_drop:
        danger_signs.append("出来高増で下落")

    if pd.notna(upside_to_20high_pct) and upside_to_20high_pct < 3:
        danger_signs.append("直近高値まで余地小")

    if high_volume_drop:
        volume_eval = "悪い"
    elif rebound_status == "あり" and pd.notna(prev5_volume) and latest_volume >= prev5_volume * 0.9:
        volume_eval = "良い"
    elif pd.notna(volume_ratio) and volume_ratio <= 1.2:
        volume_eval = "良い"
    elif pd.notna(volume_ratio) and volume_ratio <= 1.8:
        volume_eval = "普通"
    else:
        volume_eval = "注意"

    danger_count = len(danger_signs)

    trend_ok = True

    if ma_period == 25:
        if pd.notna(latest_ma75) and latest_close < latest_ma75:
            trend_ok = False

    if ma_period == 75:
        if pd.notna(latest_ma25) and pd.notna(latest_ma75):
            if latest_ma25 < latest_ma75 * 0.98:
                trend_ok = False

    if ma_period == 200:
        if pd.notna(latest_ma75) and pd.notna(latest_ma200):
            if latest_ma75 < latest_ma200 * 0.98:
                trend_ok = False
        if pd.notna(latest_ma25) and pd.notna(latest_ma75):
            if latest_ma25 < latest_ma75 * 0.95:
                trend_ok = False

    upside_ok = pd.isna(upside_to_20high_pct) or upside_to_20high_pct >= 5

    if (
        trend_ok
        and danger_count == 0
        and rebound_status == "あり"
        and volume_eval in ["良い", "普通"]
        and upside_ok
        and slope_pct > 0
    ):
        priority = "A"
        comment = "優先確認。トレンド・反発・出来高のバランスが良い"
    elif (
        danger_count <= 1
        and rebound_status in ["あり", "弱め"]
        and volume_eval != "悪い"
        and slope_pct >= 0
    ):
        priority = "B"
        comment = "監視候補。買うならチャートで反発継続を確認"
    else:
        priority = "C"
        comment = "見送り寄り。反発不足または危険サインあり"

    return {
        "優先度": priority,
        "反発確認": rebound_status,
        "危険サイン": "なし" if len(danger_signs) == 0 else " / ".join(danger_signs),
        "危険サイン数": danger_count,
        "出来高評価": volume_eval,
        "コメント": comment,
    }


def make_trade_plan(
    df,
    ma_period,
    latest_close,
    latest_ma25,
    latest_ma75,
    latest_ma200,
    priority,
    rebound_status,
    danger_count,
    danger_text,
    volume_eval
):
    """
    実践用の売買目安を作る。
    買値は「直近5日高値の上抜け」、利確は20日高値・60日高値を使う。
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    breakout_line = high.tail(5).max()

    if ma_period == 25:
        recent_low = low.tail(5).min()
        ma_stop = latest_ma25 * 0.985 if pd.notna(latest_ma25) else np.nan
    elif ma_period == 75:
        recent_low = low.tail(10).min()
        ma_stop = latest_ma75 * 0.98 if pd.notna(latest_ma75) else np.nan
    elif ma_period == 200:
        recent_low = low.tail(20).min()
        ma_stop = latest_ma200 * 0.97 if pd.notna(latest_ma200) else np.nan
    else:
        recent_low = low.tail(10).min()
        ma_stop = np.nan

    if pd.notna(recent_low) and pd.notna(ma_stop):
        stop_line = min(recent_low, ma_stop)
    elif pd.notna(recent_low):
        stop_line = recent_low
    else:
        stop_line = ma_stop

    take_profit_1 = high.tail(20).max()
    take_profit_2 = high.tail(60).max()

    breakout_to_now_pct = (breakout_line / latest_close - 1) * 100 if latest_close > 0 else np.nan

    risk = breakout_line - stop_line if pd.notna(stop_line) else np.nan
    reward1 = take_profit_1 - breakout_line if pd.notna(take_profit_1) else np.nan
    reward2 = take_profit_2 - breakout_line if pd.notna(take_profit_2) else np.nan

    loss_pct = (risk / breakout_line) * 100 if pd.notna(risk) and breakout_line > 0 else np.nan
    rr1 = reward1 / risk if pd.notna(reward1) and pd.notna(risk) and risk > 0 else np.nan
    rr2 = reward2 / risk if pd.notna(reward2) and pd.notna(risk) and risk > 0 else np.nan

    reasons = []

    if priority == "C":
        reasons.append("優先度C")

    if rebound_status == "待ち":
        reasons.append("反発確認待ち")
    elif rebound_status == "弱め":
        reasons.append("反発確認弱め")

    if danger_count >= 2:
        reasons.append("危険サイン複数")
    elif danger_text != "なし":
        reasons.append(danger_text)

    if volume_eval == "悪い":
        reasons.append("出来高評価悪い")
    elif volume_eval == "注意":
        reasons.append("出来高注意")

    if pd.isna(rr1) or rr1 < 1.0:
        if pd.isna(rr2) or rr2 < 1.5:
            reasons.append("RR悪い")
        else:
            reasons.append("第1RR低め")

    if pd.notna(loss_pct):
        if loss_pct > 10:
            reasons.append("損切り遠すぎ")
        elif loss_pct > 7:
            reasons.append("損切り遠い")

    if pd.notna(breakout_to_now_pct) and breakout_to_now_pct > 5:
        reasons.append("上抜けライン遠い")

    if (
        priority == "A"
        and rebound_status == "あり"
        and volume_eval in ["良い", "普通"]
        and danger_count == 0
        and pd.notna(rr1)
        and rr1 >= 1.5
        and pd.notna(loss_pct)
        and loss_pct <= 5
    ):
        buy_judge = "候補"
        skip_reason = "なし"
    elif (
        priority in ["A", "B"]
        and rebound_status in ["あり", "弱め"]
        and volume_eval != "悪い"
        and danger_count <= 1
        and pd.notna(loss_pct)
        and loss_pct <= 7
        and (
            (pd.notna(rr1) and rr1 >= 1.0)
            or (pd.notna(rr2) and rr2 >= 1.5)
        )
    ):
        buy_judge = "監視"
        skip_reason = " / ".join(reasons) if reasons else "買い候補に近いが最終確認必要"
    else:
        buy_judge = "見送り"
        skip_reason = " / ".join(reasons) if reasons else "条件不足"

    return {
        "買い候補": buy_judge,
        "上抜けライン": round(breakout_line, 1) if pd.notna(breakout_line) else np.nan,
        "損切りライン": round(stop_line, 1) if pd.notna(stop_line) else np.nan,
        "第1利確ライン": round(take_profit_1, 1) if pd.notna(take_profit_1) else np.nan,
        "第2利確ライン": round(take_profit_2, 1) if pd.notna(take_profit_2) else np.nan,
        "上抜けまで%": round(breakout_to_now_pct, 2) if pd.notna(breakout_to_now_pct) else np.nan,
        "想定損失%": round(loss_pct, 2) if pd.notna(loss_pct) else np.nan,
        "第1RR": round(rr1, 2) if pd.notna(rr1) else np.nan,
        "第2RR": round(rr2, 2) if pd.notna(rr2) else np.nan,
        "見送り理由": skip_reason,
    }


def detect_volume_spike_first_pullback_base(df, params):
    """
    25日線初押しの起点を探す。

    条件イメージ：
    1. 直近で出来高が前日比5倍以上に急増
    2. その後、短期的に数日上昇
    3. 調整が入り、現在株価が25日線付近
    4. 急騰後、25日線への初押しに近い
    """
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    ma25 = df["MA25"].astype(float)
    ma75 = df["MA75"].astype(float)

    latest_close = close.iloc[-1]
    latest_ma25 = ma25.iloc[-1]

    if pd.isna(latest_ma25) or latest_ma25 <= 0:
        return None

    latest_dist_ma25_pct = (latest_close / latest_ma25 - 1) * 100

    if not (params["near_lower"] <= latest_dist_ma25_pct <= params["near_upper"]):
        return None

    slope_lookback = params["ma25_slope_lookback"]
    if len(ma25.dropna()) <= slope_lookback:
        return None

    ma25_now = ma25.iloc[-1]
    ma25_ago = ma25.iloc[-1 - slope_lookback]
    ma25_slope_pct = (ma25_now / ma25_ago - 1) * 100 if ma25_ago > 0 else np.nan

    if pd.isna(ma25_slope_pct) or ma25_slope_pct < params["ma25_slope_min"]:
        return None

    # 直近の出来高急増日を探す。最低でも数日経過している必要がある。
    earliest = max(1, len(df) - params["spike_lookback"])
    latest_allowed = len(df) - params["min_days_since_spike"]

    if latest_allowed <= earliest:
        return None

    candidates = []

    for spike_pos in range(earliest, latest_allowed):
        prev_vol = volume.iloc[spike_pos - 1]
        spike_vol = volume.iloc[spike_pos]

        if prev_vol <= 0:
            continue

        volume_spike_ratio = spike_vol / prev_vol

        if volume_spike_ratio < params["volume_spike_mult"]:
            continue

        # 出来高急増日があまりにも弱い陰線なら除外しやすくする
        spike_close = close.iloc[spike_pos]
        spike_open = open_.iloc[spike_pos]
        spike_prev_close = close.iloc[spike_pos - 1]
        spike_high = high.iloc[spike_pos]
        spike_low = low.iloc[spike_pos]

        spike_up_day = spike_close > spike_prev_close
        spike_bullish = spike_close >= spike_open
        spike_close_position = (
            (spike_close - spike_low) / (spike_high - spike_low)
            if spike_high > spike_low else 0.5
        )

        if not spike_up_day:
            continue

        # 出来高急増後、短期的に数日上昇したかを見る
        rise_end = min(len(df) - 1, spike_pos + params["rise_window"])
        if rise_end <= spike_pos + 1:
            continue

        after_high = high.iloc[spike_pos + 1:rise_end + 1]
        if after_high.empty:
            continue

        peak_pos = int(after_high.values.argmax()) + spike_pos + 1
        peak_price = high.iloc[peak_pos]
        peak_date = df.index[peak_pos]
        days_to_peak = peak_pos - spike_pos

        rise_pct = (peak_price / spike_close - 1) * 100 if spike_close > 0 else np.nan

        if pd.isna(rise_pct) or rise_pct < params["min_rise_pct"]:
            continue

        # いまはピークから調整して25日線付近まで戻っているか
        pullback_pct = (peak_price / latest_close - 1) * 100 if latest_close > 0 else np.nan

        if pd.isna(pullback_pct) or pullback_pct < params["min_pullback_pct"]:
            continue

        # 初押し判定：ピーク後、現在付近の数日を除き、25日線付近まで戻っていないこと
        # 直近2日は「今まさに25日線付近に来ている」扱いとして許容する。
        first_touch_ok = True
        pre_current_end = max(peak_pos + 1, len(df) - 3)
        post_peak_before_now = df.iloc[peak_pos + 1:pre_current_end]

        if not post_peak_before_now.empty:
            dist_before = (post_peak_before_now["Close"].astype(float) / post_peak_before_now["MA25"].astype(float) - 1) * 100
            # 上から25日線に近づいた日がすでにあるなら、初押しではないとみなす
            if (dist_before <= params["first_touch_upper"]).any():
                first_touch_ok = False

        if not first_touch_ok:
            continue

        # 急騰局面で一度は25日線からしっかり上に離れていたか
        between_spike_peak = df.iloc[spike_pos:peak_pos + 1]
        max_dist_from_ma25 = (
            (between_spike_peak["Close"].astype(float) / between_spike_peak["MA25"].astype(float) - 1) * 100
        ).max()

        if pd.isna(max_dist_from_ma25) or max_dist_from_ma25 < params["near_upper"] + 2:
            continue

        # 調整感：直近で下落日がある、または5日高値から現在値が下がっている
        recent5_close = close.tail(5)
        recent_down_days = int((recent5_close.diff() < 0).sum())
        recent5_high = high.tail(5).max()
        correction_from_recent5_high_pct = (recent5_high / latest_close - 1) * 100 if latest_close > 0 else 0
        correction_ok = recent_down_days >= 1 or correction_from_recent5_high_pct >= 2

        if not correction_ok:
            continue

        candidates.append({
            "spike_pos": spike_pos,
            "spike_date": df.index[spike_pos],
            "spike_close": spike_close,
            "spike_high": spike_high,
            "spike_low": spike_low,
            "volume_spike_ratio": volume_spike_ratio,
            "spike_bullish": bool(spike_bullish),
            "spike_close_position": spike_close_position,
            "peak_pos": peak_pos,
            "peak_date": peak_date,
            "peak_price": peak_price,
            "days_to_peak": days_to_peak,
            "rise_pct": rise_pct,
            "pullback_pct": pullback_pct,
            "max_dist_from_ma25": max_dist_from_ma25,
            "recent_down_days": recent_down_days,
            "correction_from_recent5_high_pct": correction_from_recent5_high_pct,
            "ma25_slope_pct": ma25_slope_pct,
            "latest_dist_ma25_pct": latest_dist_ma25_pct,
        })

    if not candidates:
        return None

    # 直近の起点日を優先しつつ、上昇率と出来高倍率が強いものを採用
    candidates = sorted(
        candidates,
        key=lambda x: (
            x["spike_pos"],
            x["rise_pct"],
            min(x["volume_spike_ratio"], 20),
        ),
        reverse=True,
    )

    return candidates[0]


def make_first_pullback_trade_plan(df, base, latest_close, latest_ma25, params):
    """
    初押し用の売買目安。
    買いは反発確認後を想定し、直近5日高値上抜けを基本にする。
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    volume = df["Volume"].astype(float)

    latest_open = open_.iloc[-1]
    latest_high = high.iloc[-1]
    latest_low = low.iloc[-1]
    prev_close = close.iloc[-2] if len(close) >= 2 else np.nan
    prev_high = high.iloc[-2] if len(high) >= 2 else np.nan
    prev5_volume = volume.iloc[-6:-1].mean() if len(volume) >= 6 else np.nan
    latest_volume = volume.iloc[-1]

    is_bullish = latest_close > latest_open
    close_up = pd.notna(prev_close) and latest_close > prev_close
    high_break_prev = pd.notna(prev_high) and latest_high > prev_high
    volume_rebound = pd.notna(prev5_volume) and prev5_volume > 0 and latest_volume >= prev5_volume * 0.9

    rebound_score = sum([bool(is_bullish), bool(close_up), bool(high_break_prev), bool(volume_rebound)])

    if rebound_score >= 3:
        rebound_status = "あり"
    elif rebound_score == 2:
        rebound_status = "弱め"
    else:
        rebound_status = "待ち"

    # 買いは25日線付近から反発確認後の短期高値上抜けを想定
    breakout_line = high.tail(5).max()

    recent_low = low.tail(10).min()
    ma_stop = latest_ma25 * 0.975 if pd.notna(latest_ma25) else np.nan

    if pd.notna(recent_low) and pd.notna(ma_stop):
        stop_line = min(recent_low, ma_stop)
    elif pd.notna(recent_low):
        stop_line = recent_low
    else:
        stop_line = ma_stop

    # 第1利確は急騰後の高値付近、第2利確は起点から高値までの値幅をもう一度乗せる
    take_profit_1 = base["peak_price"]
    first_wave_width = max(base["peak_price"] - base["spike_close"], 0)
    take_profit_2 = base["peak_price"] + first_wave_width * 0.5

    breakout_to_now_pct = (breakout_line / latest_close - 1) * 100 if latest_close > 0 else np.nan
    risk = breakout_line - stop_line if pd.notna(stop_line) else np.nan
    reward1 = take_profit_1 - breakout_line if pd.notna(take_profit_1) else np.nan
    reward2 = take_profit_2 - breakout_line if pd.notna(take_profit_2) else np.nan

    loss_pct = (risk / breakout_line) * 100 if pd.notna(risk) and breakout_line > 0 else np.nan
    rr1 = reward1 / risk if pd.notna(reward1) and pd.notna(risk) and risk > 0 else np.nan
    rr2 = reward2 / risk if pd.notna(reward2) and pd.notna(risk) and risk > 0 else np.nan

    danger_signs = []

    if pd.notna(loss_pct) and loss_pct > params["max_loss_pct"]:
        danger_signs.append("損切り遠い")

    if pd.notna(breakout_to_now_pct) and breakout_to_now_pct > 5:
        danger_signs.append("上抜けライン遠い")

    if pd.notna(rr1) and rr1 < 1.0 and (pd.isna(rr2) or rr2 < 1.5):
        danger_signs.append("RR不足")

    if base["spike_close_position"] < 0.45:
        danger_signs.append("起点日の終値位置が弱い")

    if latest_close < latest_ma25 * 0.97:
        danger_signs.append("25日線を明確に割れ")

    if rebound_status == "あり" and not danger_signs and pd.notna(rr1) and rr1 >= 1.2 and pd.notna(loss_pct) and loss_pct <= params["max_loss_pct"]:
        buy_judge = "候補"
    elif rebound_status in ["あり", "弱め"] and len(danger_signs) <= 1 and pd.notna(loss_pct) and loss_pct <= params["max_loss_pct"] + 2:
        buy_judge = "監視"
    else:
        buy_judge = "見送り"

    skip_reason = "なし" if not danger_signs else " / ".join(danger_signs)

    if rebound_status == "待ち":
        skip_reason = "反発確認待ち" if skip_reason == "なし" else skip_reason + " / 反発確認待ち"
    elif rebound_status == "弱め" and buy_judge != "候補":
        skip_reason = "反発確認弱め" if skip_reason == "なし" else skip_reason + " / 反発確認弱め"

    return {
        "買い候補": buy_judge,
        "反発確認": rebound_status,
        "危険サイン": "なし" if not danger_signs else " / ".join(danger_signs),
        "危険サイン数": len(danger_signs),
        "上抜けライン": round(breakout_line, 1) if pd.notna(breakout_line) else np.nan,
        "損切りライン": round(stop_line, 1) if pd.notna(stop_line) else np.nan,
        "第1利確ライン": round(take_profit_1, 1) if pd.notna(take_profit_1) else np.nan,
        "第2利確ライン": round(take_profit_2, 1) if pd.notna(take_profit_2) else np.nan,
        "上抜けまで%": round(breakout_to_now_pct, 2) if pd.notna(breakout_to_now_pct) else np.nan,
        "想定損失%": round(loss_pct, 2) if pd.notna(loss_pct) else np.nan,
        "第1RR": round(rr1, 2) if pd.notna(rr1) else np.nan,
        "第2RR": round(rr2, 2) if pd.notna(rr2) else np.nan,
        "見送り理由": skip_reason,
    }


def analyze_25ma_first_pullback(
    price_df,
    code,
    name,
    market,
    params,
    use_ma200_filter=False
):
    """
    新しい25日線押し目ロジック。

    ・直近で出来高が前日比5倍以上に急増
    ・その後、短期的に数日上昇
    ・調整が入り、現在株価が25日線付近
    ・急騰後の初押しに近いものだけ拾う
    """
    if price_df.empty or len(price_df) < 100:
        return None

    df = price_df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    df["MA25"] = close.rolling(25).mean()
    df["MA75"] = close.rolling(75).mean()
    df["MA200"] = close.rolling(200).mean()

    latest_date = df.index[-1]
    latest_close = close.iloc[-1]
    latest_ma25 = df["MA25"].iloc[-1]
    latest_ma75 = df["MA75"].iloc[-1]
    latest_ma200 = df["MA200"].iloc[-1]

    if pd.isna(latest_ma25):
        return None

    if use_ma200_filter:
        if pd.isna(latest_ma200) or latest_close < latest_ma200:
            return None

    base = detect_volume_spike_first_pullback_base(df, params)

    if base is None:
        return None

    trade_plan = make_first_pullback_trade_plan(
        df=df,
        base=base,
        latest_close=latest_close,
        latest_ma25=latest_ma25,
        params=params,
    )

    def dist_to_ma(ma_value):
        if pd.notna(ma_value) and ma_value > 0:
            return (latest_close / ma_value - 1) * 100
        return np.nan

    dist_ma25_pct = dist_to_ma(latest_ma25)
    dist_ma75_pct = dist_to_ma(latest_ma75)
    dist_ma200_pct = dist_to_ma(latest_ma200)

    vol5 = volume.tail(5).mean()
    vol20 = volume.tail(20).mean()
    volume_ratio = vol5 / vol20 if vol20 > 0 else np.nan

    # 優先度
    if trade_plan["買い候補"] == "候補":
        priority = "A"
        comment = "出来高急増後の上昇から、25日線への初押し候補。反発確認あり"
    elif trade_plan["買い候補"] == "監視":
        priority = "B"
        comment = "初押し候補。買うなら反発継続と上抜けライン突破を確認"
    else:
        priority = "C"
        comment = "条件は近いが、反発・RR・損切り幅のどれかに注意"

    # スコアは並び替え用
    score = (
        min(base["volume_spike_ratio"], 20) * 1.5
        + base["rise_pct"] * 0.8
        - abs(dist_ma25_pct) * 1.2
        - max(0, base["pullback_pct"] - 18) * 0.3
        + (5 if trade_plan["反発確認"] == "あり" else 2 if trade_plan["反発確認"] == "弱め" else 0)
    )

    return {
        "コード": code,
        "銘柄名": name,
        "市場": market,
        "判定": "出来高急増後の25日線初押し",
        "買い候補": trade_plan["買い候補"],
        "上抜けライン": trade_plan["上抜けライン"],
        "損切りライン": trade_plan["損切りライン"],
        "第1利確ライン": trade_plan["第1利確ライン"],
        "第2利確ライン": trade_plan["第2利確ライン"],
        "上抜けまで%": trade_plan["上抜けまで%"],
        "想定損失%": trade_plan["想定損失%"],
        "第1RR": trade_plan["第1RR"],
        "第2RR": trade_plan["第2RR"],
        "見送り理由": trade_plan["見送り理由"],
        "優先度": priority,
        "反発確認": trade_plan["反発確認"],
        "危険サイン": trade_plan["危険サイン"],
        "出来高評価": "急増起点あり",
        "コメント": comment,
        "最新日": latest_date.strftime("%Y-%m-%d"),
        "終値": round(latest_close, 1),
        "25日線": round(latest_ma25, 1) if pd.notna(latest_ma25) else np.nan,
        "75日線": round(latest_ma75, 1) if pd.notna(latest_ma75) else np.nan,
        "200日線": round(latest_ma200, 1) if pd.notna(latest_ma200) else np.nan,
        "25日線距離%": round(dist_ma25_pct, 2) if pd.notna(dist_ma25_pct) else np.nan,
        "75日線距離%": round(dist_ma75_pct, 2) if pd.notna(dist_ma75_pct) else np.nan,
        "200日線距離%": round(dist_ma200_pct, 2) if pd.notna(dist_ma200_pct) else np.nan,
        "起点日": base["spike_date"].strftime("%Y-%m-%d"),
        "起点出来高倍率": round(base["volume_spike_ratio"], 2),
        "起点日終値": round(base["spike_close"], 1),
        "起点日終値位置": round(base["spike_close_position"], 2),
        "高値日": base["peak_date"].strftime("%Y-%m-%d"),
        "起点後高値": round(base["peak_price"], 1),
        "高値まで日数": int(base["days_to_peak"]),
        "起点後上昇率%": round(base["rise_pct"], 2),
        "高値から調整%": round(base["pullback_pct"], 2),
        "最大25日線乖離%": round(base["max_dist_from_ma25"], 2),
        "25日線傾き%": round(base["ma25_slope_pct"], 2),
        "5日出来高/20日出来高": round(volume_ratio, 2) if pd.notna(volume_ratio) else np.nan,
        "スコア": round(score, 2),
        "Yahooチャート": f"https://finance.yahoo.co.jp/quote/{code}.T/chart",
    }


def analyze_ma_pullback(
    price_df,
    code,
    name,
    market,
    ma_period,
    params,
    use_volume_filter=True,
    use_ma200_filter=False
):
    if price_df.empty or len(price_df) < ma_period + 30:
        return None

    df = price_df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    volume = df["Volume"].astype(float)

    ma_col = f"MA{ma_period}"

    df["MA25"] = close.rolling(25).mean()
    df["MA75"] = close.rolling(75).mean()
    df["MA200"] = close.rolling(200).mean()

    target_ma = df[ma_col].dropna()

    if len(target_ma) <= params["slope_lookback"]:
        return None

    latest_date = df.index[-1]
    latest_close = close.iloc[-1]
    latest_ma = df[ma_col].iloc[-1]
    latest_ma25 = df["MA25"].iloc[-1]
    latest_ma75 = df["MA75"].iloc[-1]
    latest_ma200 = df["MA200"].iloc[-1]

    if pd.isna(latest_ma):
        return None

    ma_ago = target_ma.iloc[-1 - params["slope_lookback"]]

    if ma_ago <= 0:
        return None

    slope_pct = (latest_ma / ma_ago - 1) * 100
    dist_ma_pct = (latest_close / latest_ma - 1) * 100

    def dist_to_ma(ma_value):
        if pd.notna(ma_value) and ma_value > 0:
            return (latest_close / ma_value - 1) * 100
        return np.nan

    dist_ma25_pct = dist_to_ma(latest_ma25)
    dist_ma75_pct = dist_to_ma(latest_ma75)
    dist_ma200_pct = dist_to_ma(latest_ma200)

    along_days = params["along_days"]
    recent_close = close.tail(along_days)
    recent_ma = df[ma_col].tail(along_days)

    if recent_ma.isna().any():
        return None

    recent_dist = (recent_close / recent_ma - 1) * 100

    near_count = recent_dist.between(
        params["near_lower"],
        params["near_upper"]
    ).sum()

    vol5 = volume.tail(5).mean()
    vol20 = volume.tail(20).mean()
    volume_ratio = vol5 / vol20 if vol20 > 0 else np.nan

    recent_20_high = high.tail(20).max()
    upside_to_20high_pct = (recent_20_high / latest_close - 1) * 100 if recent_20_high > 0 else np.nan

    if slope_pct < params["slope_min"]:
        return None

    if not (params["near_lower"] <= dist_ma_pct <= params["near_upper"]):
        return None

    if near_count < params["min_near_days"]:
        return None

    if use_volume_filter:
        if pd.isna(volume_ratio) or volume_ratio > params["volume_ratio_max"]:
            return None

    if use_ma200_filter:
        if pd.isna(latest_ma200) or latest_close < latest_ma200:
            return None

    score = (
        slope_pct * 2
        - abs(dist_ma_pct) * 1.2
        + near_count * 1.5
        - max(0, volume_ratio - 1) * 1.0
    )

    extra = evaluate_extra_signals(
        df=df,
        ma_period=ma_period,
        latest_close=latest_close,
        latest_ma=latest_ma,
        latest_ma25=latest_ma25,
        latest_ma75=latest_ma75,
        latest_ma200=latest_ma200,
        dist_ma_pct=dist_ma_pct,
        slope_pct=slope_pct,
        volume_ratio=volume_ratio,
        upside_to_20high_pct=upside_to_20high_pct
    )

    trade_plan = make_trade_plan(
        df=df,
        ma_period=ma_period,
        latest_close=latest_close,
        latest_ma25=latest_ma25,
        latest_ma75=latest_ma75,
        latest_ma200=latest_ma200,
        priority=extra["優先度"],
        rebound_status=extra["反発確認"],
        danger_count=extra["危険サイン数"],
        danger_text=extra["危険サイン"],
        volume_eval=extra["出来高評価"]
    )

    return {
        "コード": code,
        "銘柄名": name,
        "市場": market,
        "判定": f"{ma_period}日線押し目",
        "買い候補": trade_plan["買い候補"],
        "上抜けライン": trade_plan["上抜けライン"],
        "損切りライン": trade_plan["損切りライン"],
        "第1利確ライン": trade_plan["第1利確ライン"],
        "第2利確ライン": trade_plan["第2利確ライン"],
        "上抜けまで%": trade_plan["上抜けまで%"],
        "想定損失%": trade_plan["想定損失%"],
        "第1RR": trade_plan["第1RR"],
        "第2RR": trade_plan["第2RR"],
        "見送り理由": trade_plan["見送り理由"],
        "優先度": extra["優先度"],
        "反発確認": extra["反発確認"],
        "危険サイン": extra["危険サイン"],
        "出来高評価": extra["出来高評価"],
        "コメント": extra["コメント"],
        "最新日": latest_date.strftime("%Y-%m-%d"),
        "終値": round(latest_close, 1),
        "25日線": round(latest_ma25, 1) if pd.notna(latest_ma25) else np.nan,
        "75日線": round(latest_ma75, 1) if pd.notna(latest_ma75) else np.nan,
        "200日線": round(latest_ma200, 1) if pd.notna(latest_ma200) else np.nan,
        f"{ma_period}日線距離%": round(dist_ma_pct, 2),
        f"{ma_period}日線傾き%": round(slope_pct, 2),
        "25日線距離%": round(dist_ma25_pct, 2) if pd.notna(dist_ma25_pct) else np.nan,
        "75日線距離%": round(dist_ma75_pct, 2) if pd.notna(dist_ma75_pct) else np.nan,
        "200日線距離%": round(dist_ma200_pct, 2) if pd.notna(dist_ma200_pct) else np.nan,
        f"{ma_period}日線付近日数": int(near_count),
        "5日出来高/20日出来高": round(volume_ratio, 2) if pd.notna(volume_ratio) else np.nan,
        "直近20日高値まで%": round(upside_to_20high_pct, 2) if pd.notna(upside_to_20high_pct) else np.nan,
        "スコア": round(score, 2),
        "Yahooチャート": f"https://finance.yahoo.co.jp/quote/{code}.T/chart",
    }



def calc_close_position(row):
    high = float(row["High"])
    low = float(row["Low"])
    close = float(row["Close"])

    if high == low:
        return 0.5

    return (close - low) / (high - low)


def analyze_triangle_breakout(
    price_df,
    code,
    name,
    market,
    params,
    use_ma200_filter=False
):
    """
    三角保ち合いからの出来高増ブレイク初動を探す。

    基本思想：
    ・直近lookback日は、今日を除いて三角保ち合いを判定
    ・今日の終値が上限ラインを上抜けたらブレイク候補
    ・まだ上抜けていないが三角条件を満たすものは監視候補
    """
    lookback = params["lookback"]

    if price_df.empty or len(price_df) < lookback + 80:
        return None

    df = price_df.copy()

    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    df["MA25"] = close.rolling(25).mean()
    df["MA75"] = close.rolling(75).mean()
    df["MA200"] = close.rolling(200).mean()

    latest = df.iloc[-1]
    latest_date = df.index[-1]
    latest_close = float(close.iloc[-1])
    latest_open = float(open_.iloc[-1])
    latest_high = float(high.iloc[-1])
    latest_low = float(low.iloc[-1])
    latest_volume = float(volume.iloc[-1])

    latest_ma25 = df["MA25"].iloc[-1]
    latest_ma75 = df["MA75"].iloc[-1]
    latest_ma200 = df["MA200"].iloc[-1]

    if pd.isna(latest_ma25) or pd.isna(latest_ma75):
        return None

    if use_ma200_filter:
        if pd.isna(latest_ma200) or latest_close < latest_ma200:
            return None

    # 今日を除いた直近期間で三角保ち合いを作る
    pattern_df = df.iloc[-lookback - 1:-1].copy()

    if len(pattern_df) < lookback:
        return None

    x = np.arange(len(pattern_df))
    pattern_high = pattern_df["High"].astype(float)
    pattern_low = pattern_df["Low"].astype(float)
    pattern_close = pattern_df["Close"].astype(float)
    pattern_volume = pattern_df["Volume"].astype(float)

    high_slope, high_intercept = np.polyfit(x, pattern_high, 1)
    low_slope, low_intercept = np.polyfit(x, pattern_low, 1)

    base_price = pattern_close.mean()
    if base_price <= 0:
        return None

    high_slope_pct_per_day = high_slope / base_price * 100
    low_slope_pct_per_day = low_slope / base_price * 100

    # 今日時点に延長した上限ライン・下限ライン
    upper_line = high_slope * lookback + high_intercept
    lower_line = low_slope * lookback + low_intercept

    first_half = pattern_df.iloc[:lookback // 2]
    second_half = pattern_df.iloc[lookback // 2:]

    first_range = first_half["High"].max() - first_half["Low"].min()
    second_range = second_half["High"].max() - second_half["Low"].min()
    range_ratio = second_range / first_range if first_range > 0 else np.nan

    first_volume = first_half["Volume"].mean()
    second_volume = second_half["Volume"].mean()
    volume_contract_ratio = second_volume / first_volume if first_volume > 0 else np.nan

    prev5_volume = volume.iloc[-6:-1].mean() if len(volume) >= 6 else np.nan
    volume_ratio = latest_volume / prev5_volume if pd.notna(prev5_volume) and prev5_volume > 0 else np.nan

    ma25_up = df["MA25"].iloc[-1] > df["MA25"].iloc[-6] if len(df) >= 6 else False
    ma75_up = df["MA75"].iloc[-1] > df["MA75"].iloc[-11] if len(df) >= 11 else False
    above_ma75 = latest_close > latest_ma75

    is_bullish = latest_close > latest_open
    close_position = calc_close_position(latest)

    range_shrinking = pd.notna(range_ratio) and range_ratio <= params["range_ratio_max"]
    volume_contracting = pd.notna(volume_contract_ratio) and volume_contract_ratio <= params["volume_contract_ratio_max"]

    triangle_shape_ok = (
        high_slope_pct_per_day <= params["high_slope_max_pct_per_day"]
        and low_slope_pct_per_day >= params["low_slope_min_pct_per_day"]
        and range_shrinking
        and volume_contracting
        and above_ma75
    )

    if params.get("need_ma75_up", False) and not ma75_up:
        triangle_shape_ok = False

    breakout_line = upper_line
    breakout_over_pct = (latest_close / breakout_line - 1) * 100 if breakout_line > 0 else np.nan
    breakout_to_now_pct = (breakout_line / latest_close - 1) * 100 if latest_close > 0 else np.nan

    is_breakout = (
        triangle_shape_ok
        and latest_close > breakout_line
        and pd.notna(volume_ratio)
        and volume_ratio >= params["breakout_volume_min"]
        and is_bullish
        and close_position >= params["close_position_min"]
    )

    # 損切りは「三角下限」と「直近10日安値」の近い方を採用。
    # 遠すぎる損切りを避けるため、直近安値を優先しやすくする。
    recent_10_low = low.tail(10).min()
    stop_line_candidates = [v for v in [lower_line, recent_10_low] if pd.notna(v)]
    stop_line = max(stop_line_candidates) if stop_line_candidates else np.nan

    # 利確目安は三角の値幅を上に出す。
    triangle_width = max(upper_line - lower_line, 0)
    take_profit_1 = latest_close + triangle_width
    take_profit_2 = latest_close + triangle_width * 1.5

    risk = latest_close - stop_line if pd.notna(stop_line) else np.nan
    reward1 = take_profit_1 - latest_close if pd.notna(take_profit_1) else np.nan
    reward2 = take_profit_2 - latest_close if pd.notna(take_profit_2) else np.nan

    loss_pct = risk / latest_close * 100 if pd.notna(risk) and latest_close > 0 else np.nan
    rr1 = reward1 / risk if pd.notna(reward1) and pd.notna(risk) and risk > 0 else np.nan
    rr2 = reward2 / risk if pd.notna(reward2) and pd.notna(risk) and risk > 0 else np.nan

    reasons = []

    if not triangle_shape_ok:
        if high_slope_pct_per_day > params["high_slope_max_pct_per_day"]:
            reasons.append("高値ラインが切り下がっていない")
        if low_slope_pct_per_day < params["low_slope_min_pct_per_day"]:
            reasons.append("安値の切り上がりが弱い")
        if not range_shrinking:
            reasons.append("値幅縮小が弱い")
        if not volume_contracting:
            reasons.append("保ち合い中の出来高減少が弱い")
        if not above_ma75:
            reasons.append("株価が75日線より下")
        if params.get("need_ma75_up", False) and not ma75_up:
            reasons.append("75日線が上向きではない")

        # 三角ですらない銘柄は結果に出さない
        return None

    if not is_breakout:
        if latest_close <= breakout_line:
            reasons.append("まだ上抜けライン未突破")
        if pd.isna(volume_ratio) or volume_ratio < params["breakout_volume_min"]:
            reasons.append("ブレイク出来高不足")
        if not is_bullish:
            reasons.append("陽線ではない")
        if close_position < params["close_position_min"]:
            reasons.append("終値位置が弱い")

    if pd.notna(loss_pct) and loss_pct > params["stop_loss_max_pct"]:
        reasons.append("損切りラインまでが遠い")

    if pd.isna(rr1) or rr1 < params["rr_min"]:
        reasons.append("RR不足")

    if is_breakout and pd.notna(loss_pct) and loss_pct <= params["stop_loss_max_pct"] and pd.notna(rr1) and rr1 >= params["rr_min"]:
        buy_judge = "候補"
        priority = "A" if volume_ratio >= params["breakout_volume_min"] * 1.5 and close_position >= 0.65 else "B"
        skip_reason = "なし"
        comment = "出来高を伴って三角保ち合いを上抜け。初動候補として優先確認"
    elif triangle_shape_ok and latest_close <= breakout_line:
        buy_judge = "監視"
        priority = "B"
        skip_reason = " / ".join(reasons) if reasons else "上抜け待ち"
        comment = "三角保ち合い中。上抜けライン突破と出来高増を待つ"
    else:
        buy_judge = "見送り"
        priority = "C"
        skip_reason = " / ".join(reasons) if reasons else "条件不足"
        comment = "形はあるが、出来高・終値位置・RRのどれかが不足"

    # スコアは並び替え用。売買判断そのものではない。
    tight_score = max(0, (1 - range_ratio) * 30) if pd.notna(range_ratio) else 0
    volume_contract_score = max(0, (1 - volume_contract_ratio) * 20) if pd.notna(volume_contract_ratio) else 0
    volume_score = min(volume_ratio, 5) * 5 if pd.notna(volume_ratio) else 0
    close_score = close_position * 10
    trend_score = (5 if ma25_up else 0) + (5 if ma75_up else 0) + (5 if above_ma75 else 0)
    risk_penalty = max(0, loss_pct - params["stop_loss_max_pct"]) * 2 if pd.notna(loss_pct) else 10
    score = tight_score + volume_contract_score + volume_score + close_score + trend_score - risk_penalty

    return {
        "コード": code,
        "銘柄名": name,
        "市場": market,
        "判定": "三角保ち合いブレイク",
        "買い候補": buy_judge,
        "上抜けライン": round(breakout_line, 1) if pd.notna(breakout_line) else np.nan,
        "損切りライン": round(stop_line, 1) if pd.notna(stop_line) else np.nan,
        "第1利確ライン": round(take_profit_1, 1) if pd.notna(take_profit_1) else np.nan,
        "第2利確ライン": round(take_profit_2, 1) if pd.notna(take_profit_2) else np.nan,
        "上抜けまで%": round(breakout_to_now_pct, 2) if pd.notna(breakout_to_now_pct) else np.nan,
        "ブレイク超過%": round(breakout_over_pct, 2) if pd.notna(breakout_over_pct) else np.nan,
        "想定損失%": round(loss_pct, 2) if pd.notna(loss_pct) else np.nan,
        "第1RR": round(rr1, 2) if pd.notna(rr1) else np.nan,
        "第2RR": round(rr2, 2) if pd.notna(rr2) else np.nan,
        "見送り理由": skip_reason,
        "優先度": priority,
        "反発確認": "あり" if is_breakout else "上抜け待ち",
        "危険サイン": "なし" if skip_reason == "なし" else skip_reason,
        "出来高評価": "良い" if pd.notna(volume_ratio) and volume_ratio >= params["breakout_volume_min"] else "不足",
        "コメント": comment,
        "最新日": latest_date.strftime("%Y-%m-%d"),
        "終値": round(latest_close, 1),
        "25日線": round(latest_ma25, 1) if pd.notna(latest_ma25) else np.nan,
        "75日線": round(latest_ma75, 1) if pd.notna(latest_ma75) else np.nan,
        "200日線": round(latest_ma200, 1) if pd.notna(latest_ma200) else np.nan,
        "保ち合い日数": lookback,
        "保ち合い上限": round(upper_line, 1) if pd.notna(upper_line) else np.nan,
        "保ち合い下限": round(lower_line, 1) if pd.notna(lower_line) else np.nan,
        "高値傾き%/日": round(high_slope_pct_per_day, 3),
        "安値傾き%/日": round(low_slope_pct_per_day, 3),
        "値幅縮小率": round(range_ratio, 2) if pd.notna(range_ratio) else np.nan,
        "出来高収縮率": round(volume_contract_ratio, 2) if pd.notna(volume_contract_ratio) else np.nan,
        "出来高倍率": round(volume_ratio, 2) if pd.notna(volume_ratio) else np.nan,
        "終値位置": round(close_position, 2),
        "25日線上向き": bool(ma25_up),
        "75日線上向き": bool(ma75_up),
        "株価75日線上": bool(above_ma75),
        "スコア": round(score, 2),
        "Yahooチャート": f"https://finance.yahoo.co.jp/quote/{code}.T/chart",
    }



def resample_ohlcv_for_cup(df, timeframe):
    """
    日足データを、週足・月足に変換する。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    rule = CUP_TIMEFRAME_SETTINGS[timeframe]["resample_rule"]

    if rule is None:
        out = df.copy()
    else:
        out = df.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def _safe_pct(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a / b - 1) * 100


def detect_cup_with_handle_one_timeframe(price_df, timeframe, params):
    """
    カップウィズハンドルの近い形を数値で探す。

    完璧な画像認識ではなく、以下を機械判定する：
    ・左リムから十分に下落してカップ底を作った
    ・右リムが左リム付近まで戻った
    ・直近で浅いハンドル調整を作った
    ・現在値がピボット近く、または軽く上抜けている
    """
    tf = CUP_TIMEFRAME_SETTINGS[timeframe]
    df = resample_ohlcv_for_cup(price_df, timeframe)

    if df.empty or len(df) < tf["min_bars"]:
        return None

    df = df.copy()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA30"] = df["Close"].rolling(30).mean()

    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    latest_date = df.index[-1]
    latest_close = close.iloc[-1]
    latest_open = open_.iloc[-1]
    latest_high = high.iloc[-1]
    latest_low = low.iloc[-1]
    latest_volume = volume.iloc[-1]

    if latest_close <= 0:
        return None

    candidates = []

    for handle_len in range(tf["handle_min"], tf["handle_max"] + 1, tf["handle_step"]):
        if len(df) < tf["cup_min"] + handle_len + 10:
            continue

        handle_df = df.iloc[-handle_len:].copy()
        pre_handle_end = len(df) - handle_len

        if pre_handle_end <= 0:
            continue

        # 右リムはハンドル開始直前付近の高値。
        right_zone_start = max(0, pre_handle_end - max(3, handle_len))
        right_zone = df.iloc[right_zone_start:pre_handle_end]
        if right_zone.empty:
            continue

        pivot = right_zone["High"].max()
        right_rim_pos = int(df.index.get_loc(right_zone["High"].idxmax()))

        if pd.isna(pivot) or pivot <= 0:
            continue

        handle_low = handle_df["Low"].min()
        handle_depth_pct = (pivot - handle_low) / pivot * 100 if pivot > 0 else np.nan

        if pd.isna(handle_depth_pct):
            continue

        if handle_depth_pct < params["min_handle_depth_pct"] or handle_depth_pct > params["max_handle_depth_pct"]:
            continue

        # 現在値はピボット近く、もしくは少し上抜けまで許容
        pivot_distance_pct = (pivot / latest_close - 1) * 100
        breakout_over_pct = (latest_close / pivot - 1) * 100

        if pivot_distance_pct > params["pivot_distance_max_pct"]:
            continue

        if breakout_over_pct > params["breakout_over_allow_pct"]:
            continue

        for cup_len in range(tf["cup_min"], tf["cup_max"] + 1, tf["cup_step"]):
            cup_start = pre_handle_end - cup_len
            if cup_start < 0:
                continue

            cup_df = df.iloc[cup_start:pre_handle_end]
            if len(cup_df) < cup_len:
                continue

            # 左リムはカップ前半の高値、底はその後の安値
            left_area = cup_df.iloc[:max(5, int(cup_len * 0.55))]
            if left_area.empty:
                continue

            left_rim_high = left_area["High"].max()
            left_rim_idx = left_area["High"].idxmax()
            left_rim_pos = int(df.index.get_loc(left_rim_idx))

            bottom_area = df.iloc[left_rim_pos + 1:pre_handle_end]
            if bottom_area.empty:
                continue

            bottom_low = bottom_area["Low"].min()
            bottom_idx = bottom_area["Low"].idxmin()
            bottom_pos = int(df.index.get_loc(bottom_idx))

            # 底が左リム直後や右リム直前すぎるものは、カップというより急落急騰扱いにする
            bottom_rel = (bottom_pos - cup_start) / cup_len
            if bottom_rel < 0.20 or bottom_rel > 0.80:
                continue

            if pd.isna(left_rim_high) or left_rim_high <= 0 or pd.isna(bottom_low):
                continue

            cup_depth_pct = (left_rim_high - bottom_low) / left_rim_high * 100

            if cup_depth_pct < params["min_cup_depth_pct"] or cup_depth_pct > params["max_cup_depth_pct"]:
                continue

            # 右リムが左リム付近まで戻っているか
            rim_diff_pct = abs(pivot / left_rim_high - 1) * 100
            if rim_diff_pct > params["rim_tolerance_pct"]:
                continue

            # ハンドルがカップに対して深すぎるものは除外
            if handle_depth_pct > cup_depth_pct * params["handle_vs_cup_max"]:
                continue

            handle_avg_volume = handle_df["Volume"].mean()
            cup_avg_volume = cup_df["Volume"].mean()
            volume_contract_ratio = handle_avg_volume / cup_avg_volume if cup_avg_volume > 0 else np.nan
            volume_contract_ok = pd.notna(volume_contract_ratio) and volume_contract_ratio <= 1.10

            if params.get("need_volume_contract", False) and not volume_contract_ok:
                continue

            ma10 = df["MA10"].iloc[-1]
            ma30 = df["MA30"].iloc[-1]
            ma_trend_ok = True
            if pd.notna(ma10) and pd.notna(ma30):
                ma_trend_ok = latest_close >= ma10 * 0.97 and ma10 >= ma30 * 0.95

            if params.get("need_ma_trend", False) and not ma_trend_ok:
                continue

            prev_volume_mean = volume.iloc[-6:-1].mean() if len(volume) >= 6 else np.nan
            volume_ratio = latest_volume / prev_volume_mean if pd.notna(prev_volume_mean) and prev_volume_mean > 0 else np.nan

            is_bullish = latest_close > latest_open
            close_position = (latest_close - latest_low) / (latest_high - latest_low) if latest_high > latest_low else 0.5
            is_breakout = latest_close > pivot and pd.notna(volume_ratio) and volume_ratio >= 1.3 and close_position >= 0.55

            # 損切りはハンドル安値少し下を想定
            stop_line = handle_low * 0.985
            risk = latest_close - stop_line
            cup_width = max(pivot - bottom_low, 0)
            take_profit_1 = pivot + cup_width * 0.50
            take_profit_2 = pivot + cup_width
            reward1 = take_profit_1 - latest_close
            reward2 = take_profit_2 - latest_close
            loss_pct = risk / latest_close * 100 if latest_close > 0 else np.nan
            rr1 = reward1 / risk if risk > 0 else np.nan
            rr2 = reward2 / risk if risk > 0 else np.nan

            reasons = []
            if latest_close <= pivot:
                reasons.append("ピボット上抜け待ち")
            if pd.notna(volume_ratio) and volume_ratio < 1.3:
                reasons.append("上抜け出来高待ち")
            if not is_bullish:
                reasons.append("陽線待ち")
            if close_position < 0.55:
                reasons.append("終値位置が弱い")
            if pd.notna(loss_pct) and loss_pct > 12:
                reasons.append("損切り遠い")
            if pd.notna(rr1) and rr1 < 1.0 and (pd.isna(rr2) or rr2 < 1.5):
                reasons.append("RR不足")
            if not volume_contract_ok:
                reasons.append("ハンドル中の出来高収縮が弱い")
            if not ma_trend_ok:
                reasons.append("移動平均の形が弱い")

            if is_breakout and (pd.isna(loss_pct) or loss_pct <= 12) and (pd.notna(rr1) and rr1 >= 1.0):
                buy_judge = "候補"
                priority = "A"
                comment = "カップウィズハンドルのピボットを出来高増で上抜け。初動候補"
                skip_reason = "なし"
            elif latest_close <= pivot and pivot_distance_pct <= params["pivot_distance_max_pct"]:
                buy_judge = "監視"
                priority = "B"
                comment = "カップウィズハンドル形成後、ピボット上抜け待ち"
                skip_reason = " / ".join(reasons) if reasons else "上抜け待ち"
            else:
                buy_judge = "見送り"
                priority = "C"
                comment = "形は近いが、出来高・終値位置・RRなどが不足"
                skip_reason = " / ".join(reasons) if reasons else "条件不足"

            # 並び替え用スコア
            distance_score = max(0, params["pivot_distance_max_pct"] - max(pivot_distance_pct, 0)) * 4
            cup_score = min(cup_depth_pct, 40) * 0.5
            handle_score = max(0, params["max_handle_depth_pct"] - handle_depth_pct) * 1.5
            rim_score = max(0, params["rim_tolerance_pct"] - rim_diff_pct) * 1.5
            vol_score = (min(volume_ratio, 3) * 5 if pd.notna(volume_ratio) else 0)
            rr_score = (min(rr1, 3) * 5 if pd.notna(rr1) else 0)
            score = distance_score + cup_score + handle_score + rim_score + vol_score + rr_score
            if buy_judge == "候補":
                score += 20
            elif buy_judge == "監視":
                score += 10

            candidates.append({
                "足種": timeframe,
                "判定": "カップウィズハンドル",
                "買い候補": buy_judge,
                "優先度": priority,
                "上抜けライン": round(pivot, 1),
                "損切りライン": round(stop_line, 1),
                "第1利確ライン": round(take_profit_1, 1),
                "第2利確ライン": round(take_profit_2, 1),
                "上抜けまで%": round(pivot_distance_pct, 2),
                "ブレイク超過%": round(breakout_over_pct, 2),
                "想定損失%": round(loss_pct, 2) if pd.notna(loss_pct) else np.nan,
                "第1RR": round(rr1, 2) if pd.notna(rr1) else np.nan,
                "第2RR": round(rr2, 2) if pd.notna(rr2) else np.nan,
                "見送り理由": skip_reason,
                "反発確認": "あり" if is_breakout else "上抜け待ち",
                "危険サイン": "なし" if skip_reason == "なし" else skip_reason,
                "出来高評価": "良い" if pd.notna(volume_ratio) and volume_ratio >= 1.3 else "待ち",
                "コメント": comment,
                "最新日": latest_date.strftime("%Y-%m-%d"),
                "終値": round(latest_close, 1),
                "左リム高値": round(left_rim_high, 1),
                "カップ底値": round(bottom_low, 1),
                "右リム/ピボット": round(pivot, 1),
                "カップ深さ%": round(cup_depth_pct, 2),
                "ハンドル深さ%": round(handle_depth_pct, 2),
                "リム差%": round(rim_diff_pct, 2),
                "カップ期間本数": int(cup_len),
                "ハンドル期間本数": int(handle_len),
                "出来高倍率": round(volume_ratio, 2) if pd.notna(volume_ratio) else np.nan,
                "ハンドル出来高/カップ出来高": round(volume_contract_ratio, 2) if pd.notna(volume_contract_ratio) else np.nan,
                "終値位置": round(close_position, 2),
                "MA10": round(ma10, 1) if pd.notna(ma10) else np.nan,
                "MA30": round(ma30, 1) if pd.notna(ma30) else np.nan,
                "スコア": round(score, 2),
            })

    if not candidates:
        return None

    buy_rank = {"候補": 0, "監視": 1, "見送り": 2}
    priority_rank = {"A": 0, "B": 1, "C": 2}
    candidates = sorted(
        candidates,
        key=lambda x: (
            buy_rank.get(x["買い候補"], 9),
            priority_rank.get(x["優先度"], 9),
            -x["スコア"],
        )
    )

    return candidates[0]


def analyze_cup_with_handle(
    price_df,
    code,
    name,
    market,
    params,
    target_timeframes,
    use_ma200_filter=False
):
    """
    指定された足種ごとにカップウィズハンドルを判定し、該当したものを複数行で返す。
    """
    if price_df.empty or len(price_df) < 120:
        return []

    if use_ma200_filter:
        close = price_df["Close"].astype(float)
        ma200 = close.rolling(200).mean().iloc[-1]
        if pd.isna(ma200) or close.iloc[-1] < ma200:
            return []

    results = []

    for timeframe in target_timeframes:
        detected = detect_cup_with_handle_one_timeframe(price_df, timeframe, params)
        if detected is None:
            continue

        detected["コード"] = code
        detected["銘柄名"] = name
        detected["市場"] = market
        detected["Yahooチャート"] = f"https://finance.yahoo.co.jp/quote/{code}.T/chart"
        results.append(detected)

    return results

def show_saved_results(ma_period):
    """
    Streamlitはselectboxを変更するたびに画面全体が再実行される。
    そのため、直前の抽出結果をsession_stateに保存してから表示する。
    """
    result_key = f"screen_result_df_{ma_period}"
    failed_key = f"screen_failed_count_{ma_period}"
    market_key = f"screen_market_{ma_period}"

    if result_key not in st.session_state:
        return

    result_df = st.session_state[result_key]
    failed_count = st.session_state.get(failed_key, 0)
    saved_market = st.session_state.get(market_key, "")

    st.subheader("抽出結果")

    if result_df is None or result_df.empty:
        st.warning("条件に一致する銘柄は見つかりませんでした。条件を「ゆるめ」にするか、確認銘柄数を増やしてください。")
        return

    st.write(f"抽出銘柄数：{len(result_df)}")
    st.write(f"取得失敗数：{failed_count}")

    judge_filter = st.selectbox(
        "表示する買い候補",
        ["すべて", "候補のみ", "候補・監視", "監視のみ", "見送りのみ"],
        key=f"judge_filter_{ma_period}"
    )

    display_df = result_df.copy()

    if judge_filter == "候補のみ":
        display_df = display_df[display_df["買い候補"] == "候補"]
    elif judge_filter == "候補・監視":
        display_df = display_df[display_df["買い候補"].isin(["候補", "監視"])]
    elif judge_filter == "監視のみ":
        display_df = display_df[display_df["買い候補"] == "監視"]
    elif judge_filter == "見送りのみ":
        display_df = display_df[display_df["買い候補"] == "見送り"]

    display_df = display_df.drop(columns=["_買い候補順", "_優先度順"], errors="ignore")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = display_df.to_csv(index=False, encoding="utf-8-sig")
    market_for_filename = str(saved_market).replace("全市場", "all_markets")

    st.download_button(
        label="CSVダウンロード",
        data=csv,
        file_name=f"ma{ma_period}_pullback_{market_for_filename}_trade_plan.csv",
        mime="text/csv",
        key=f"download_ma{ma_period}_{market_for_filename}"
    )

def run_screener(
    ma_period,
    market_choice,
    preset_name,
    max_scan,
    wait_sec,
    use_volume_filter,
    use_ma200_filter
):
    params = PRESETS[ma_period][preset_name]

    try:
        all_stocks = load_jpx_list()
    except Exception as e:
        st.error(f"JPXの銘柄リスト取得に失敗しました: {e}")
        st.stop()

    target_stocks = filter_market(all_stocks, market_choice)

    if target_stocks.empty:
        st.warning("対象銘柄が見つかりませんでした。")
        st.stop()

    total_all = len(target_stocks)

    if max_scan > 0:
        target_stocks = target_stocks.head(max_scan)

    total = len(target_stocks)

    st.write(f"対象市場：{market_choice}")
    st.write(f"対象銘柄数：{total} / 選択市場全体 {total_all}")

    progress_bar = st.progress(0)
    status_area = st.empty()

    results = []
    failed_count = 0

    for n, (_, row) in enumerate(target_stocks.iterrows(), start=1):
        code = row["コード"]
        name = row["銘柄名"]
        market = row["市場"]

        status_area.write(f"{n}/{total} 確認中：{code} {name}")

        try:
            price_df = download_price(code)

            if ma_period == 25:
                result = analyze_25ma_first_pullback(
                    price_df=price_df,
                    code=code,
                    name=name,
                    market=market,
                    params=params,
                    use_ma200_filter=use_ma200_filter
                )
            else:
                result = analyze_ma_pullback(
                    price_df=price_df,
                    code=code,
                    name=name,
                    market=market,
                    ma_period=ma_period,
                    params=params,
                    use_volume_filter=use_volume_filter,
                    use_ma200_filter=use_ma200_filter
                )

            if result is not None:
                result["33業種"] = row.get("33業種", "")
                result["17業種"] = row.get("17業種", "")
                result["テーマ"] = row.get("テーマ", infer_theme(code, name, row.get("33業種", ""), row.get("17業種", "")))
                results.append(result)

        except Exception:
            failed_count += 1

        progress_bar.progress(n / total)

        if wait_sec > 0:
            time.sleep(wait_sec)

    status_area.write("スクリーニング完了")

    result_key = f"screen_result_df_{ma_period}"
    failed_key = f"screen_failed_count_{ma_period}"
    market_key = f"screen_market_{ma_period}"

    if len(results) == 0:
        st.session_state[result_key] = pd.DataFrame()
        st.session_state[failed_key] = failed_count
        st.session_state[market_key] = market_choice
    else:
        result_df = pd.DataFrame(results)

        buy_rank = {"候補": 0, "監視": 1, "見送り": 2}
        priority_rank = {"A": 0, "B": 1, "C": 2}
        result_df["_買い候補順"] = result_df["買い候補"].map(buy_rank).fillna(9)
        result_df["_優先度順"] = result_df["優先度"].map(priority_rank).fillna(9)

        sort_cols = ["_買い候補順", "_優先度順", "第1RR", "第2RR", "スコア"]
        result_df = result_df.sort_values(
            by=sort_cols,
            ascending=[True, True, False, False, False]
        ).reset_index(drop=True)

        st.session_state[result_key] = result_df
        st.session_state[failed_key] = failed_count
        st.session_state[market_key] = market_choice

    show_saved_results(ma_period)


def render_tab(ma_period, key_prefix):
    if ma_period == 25:
        st.subheader("出来高急増後の25日線初押しスクリーニング")
        st.write("前日比5倍以上の出来高急増を起点に、数日上昇した後、初めて25日線付近まで押している銘柄を探します。")
    elif ma_period == 75:
        st.subheader("75日線押し目スクリーニング")
        st.write("中期上昇トレンド中に、75日線まで深めに押している銘柄を探します。")
    elif ma_period == 200:
        st.subheader("200日線押し目スクリーニング")
        st.write("長期上昇トレンド中に、200日線付近まで大きく押した銘柄を探します。大底狙いではなく、200日線が上向き〜横ばいで、反発の兆しがあるものを優先します。")
    else:
        st.subheader(f"{ma_period}日線押し目スクリーニング")
        st.write(f"{ma_period}日線付近まで押している銘柄を探します。")

    col1, col2 = st.columns(2)

    with col1:
        preset_options = [x for x in ["普通", "厳しめ", "超厳しめ", "ゆるめ"] if x in PRESETS[ma_period]]
        default_index = preset_options.index("厳しめ") if "厳しめ" in preset_options else 0
        preset_name = st.selectbox(
            "判定の厳しさ",
            preset_options,
            index=default_index,
            key=f"{key_prefix}_preset"
        )

    with col2:
        params = PRESETS[ma_period][preset_name]
        st.write("現在の条件")
        if ma_period == 25:
            st.write(f"出来高急増：前日比 {params['volume_spike_mult']}倍以上")
            st.write(f"起点日探索：直近{params['spike_lookback']}営業日")
            st.write(f"起点後上昇率：{params['min_rise_pct']}%以上")
            st.write(f"現在の25日線距離：{params['near_lower']}% 〜 +{params['near_upper']}%")
            st.write(f"高値からの調整：{params['min_pullback_pct']}%以上")
            st.write("初押し判定：高値後、今回が最初の25日線接近に近いもの")
        else:
            st.write(f"{ma_period}日線との距離：{params['near_lower']}% 〜 +{params['near_upper']}%")
            st.write(f"{ma_period}日線の傾き：{params['slope_min']}%以上")
            st.write(f"直近{params['along_days']}日中、{params['min_near_days']}日以上が{ma_period}日線付近")

    button_label = "25日線初押しスクリーニング実行" if ma_period == 25 else f"{ma_period}日線スクリーニング実行"

    run_button = st.button(
        button_label,
        type="primary",
        key=f"{key_prefix}_run"
    )

    if run_button:
        run_screener(
            ma_period=ma_period,
            market_choice=market_choice,
            preset_name=preset_name,
            max_scan=max_scan,
            wait_sec=wait_sec,
            use_volume_filter=use_volume_filter,
            use_ma200_filter=use_ma200_filter
        )
    else:
        show_saved_results(ma_period)


def show_saved_triangle_results():
    """
    三角保ち合いブレイク用の保存結果表示。
    """
    result_key = "screen_result_df_triangle"
    failed_key = "screen_failed_count_triangle"
    market_key = "screen_market_triangle"

    if result_key not in st.session_state:
        return

    result_df = st.session_state[result_key]
    failed_count = st.session_state.get(failed_key, 0)
    saved_market = st.session_state.get(market_key, "")

    st.subheader("抽出結果")

    if result_df is None or result_df.empty:
        st.warning("条件に一致する銘柄は見つかりませんでした。条件を「ゆるめ」にするか、確認銘柄数を増やしてください。")
        return

    st.write(f"抽出銘柄数：{len(result_df)}")
    st.write(f"取得失敗数：{failed_count}")

    judge_filter = st.selectbox(
        "表示する買い候補",
        ["すべて", "候補のみ", "候補・監視", "監視のみ", "見送りのみ"],
        key="judge_filter_triangle"
    )

    display_df = result_df.copy()

    if judge_filter == "候補のみ":
        display_df = display_df[display_df["買い候補"] == "候補"]
    elif judge_filter == "候補・監視":
        display_df = display_df[display_df["買い候補"].isin(["候補", "監視"])]
    elif judge_filter == "監視のみ":
        display_df = display_df[display_df["買い候補"] == "監視"]
    elif judge_filter == "見送りのみ":
        display_df = display_df[display_df["買い候補"] == "見送り"]

    display_df = display_df.drop(columns=["_買い候補順", "_優先度順"], errors="ignore")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = display_df.to_csv(index=False, encoding="utf-8-sig")
    market_for_filename = str(saved_market).replace("全市場", "all_markets")

    st.download_button(
        label="CSVダウンロード",
        data=csv,
        file_name=f"triangle_breakout_{market_for_filename}_trade_plan.csv",
        mime="text/csv",
        key=f"download_triangle_{market_for_filename}"
    )


def run_triangle_screener(
    market_choice,
    preset_name,
    max_scan,
    wait_sec,
    use_ma200_filter
):
    params = TRIANGLE_PRESETS[preset_name]

    try:
        all_stocks = load_jpx_list()
    except Exception as e:
        st.error(f"JPXの銘柄リスト取得に失敗しました: {e}")
        st.stop()

    target_stocks = filter_market(all_stocks, market_choice)

    if target_stocks.empty:
        st.warning("対象銘柄が見つかりませんでした。")
        st.stop()

    total_all = len(target_stocks)

    if max_scan > 0:
        target_stocks = target_stocks.head(max_scan)

    total = len(target_stocks)

    st.write(f"対象市場：{market_choice}")
    st.write(f"対象銘柄数：{total} / 選択市場全体 {total_all}")

    progress_bar = st.progress(0)
    status_area = st.empty()

    results = []
    failed_count = 0

    for n, (_, row) in enumerate(target_stocks.iterrows(), start=1):
        code = row["コード"]
        name = row["銘柄名"]
        market = row["市場"]

        status_area.write(f"{n}/{total} 確認中：{code} {name}")

        try:
            price_df = download_price(code)

            result = analyze_triangle_breakout(
                price_df=price_df,
                code=code,
                name=name,
                market=market,
                params=params,
                use_ma200_filter=use_ma200_filter
            )

            if result is not None:
                result["33業種"] = row.get("33業種", "")
                result["17業種"] = row.get("17業種", "")
                result["テーマ"] = row.get("テーマ", infer_theme(code, name, row.get("33業種", ""), row.get("17業種", "")))
                results.append(result)

        except Exception:
            failed_count += 1

        progress_bar.progress(n / total)

        if wait_sec > 0:
            time.sleep(wait_sec)

    status_area.write("スクリーニング完了")

    result_key = "screen_result_df_triangle"
    failed_key = "screen_failed_count_triangle"
    market_key = "screen_market_triangle"

    if len(results) == 0:
        st.session_state[result_key] = pd.DataFrame()
        st.session_state[failed_key] = failed_count
        st.session_state[market_key] = market_choice
    else:
        result_df = pd.DataFrame(results)

        buy_rank = {"候補": 0, "監視": 1, "見送り": 2}
        priority_rank = {"A": 0, "B": 1, "C": 2}
        result_df["_買い候補順"] = result_df["買い候補"].map(buy_rank).fillna(9)
        result_df["_優先度順"] = result_df["優先度"].map(priority_rank).fillna(9)

        sort_cols = ["_買い候補順", "_優先度順", "第1RR", "第2RR", "出来高倍率", "スコア"]
        result_df = result_df.sort_values(
            by=sort_cols,
            ascending=[True, True, False, False, False, False]
        ).reset_index(drop=True)

        st.session_state[result_key] = result_df
        st.session_state[failed_key] = failed_count
        st.session_state[market_key] = market_choice

    show_saved_triangle_results()


def render_triangle_tab(key_prefix):
    st.subheader("三角保ち合いブレイク初動スクリーニング")

    st.write(
        "トライアル・住友ファーマ・ワークマンのように、"
        "上昇トレンド中の三角保ち合いから出来高増で上抜ける初動候補を探します。"
    )

    col1, col2 = st.columns(2)

    with col1:
        preset_name = st.selectbox(
            "判定の厳しさ",
            ["普通", "厳しめ", "超厳しめ", "ゆるめ"],
            index=1,
            key=f"{key_prefix}_preset"
        )

    with col2:
        params = TRIANGLE_PRESETS[preset_name]
        st.write("現在の条件")
        st.write(f"保ち合い期間：直近{params['lookback']}営業日")
        st.write(f"ブレイク出来高：直近5日平均の{params['breakout_volume_min']}倍以上")
        st.write(f"値幅縮小率：{params['range_ratio_max']}以下")
        st.write(f"損切り許容：{params['stop_loss_max_pct']}%以内")

    run_button = st.button(
        "三角保ち合いブレイクをスクリーニング実行",
        type="primary",
        key=f"{key_prefix}_run"
    )

    if run_button:
        run_triangle_screener(
            market_choice=market_choice,
            preset_name=preset_name,
            max_scan=max_scan,
            wait_sec=wait_sec,
            use_ma200_filter=use_ma200_filter
        )
    else:
        show_saved_triangle_results()




def analyze_momentum(price_df, code, name, market, sector33, sector17, theme, params, use_ma200_filter):
    """
    モメンタム初動・強勢継続をスコア化する。
    強い銘柄に素直についていく相場用。
    """
    if price_df is None or price_df.empty or len(price_df) < 90:
        return None

    df = price_df.copy()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA25"] = df["Close"].rolling(25).mean()
    df["MA75"] = df["Close"].rolling(75).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    latest_close = close.iloc[-1]
    latest_open = open_.iloc[-1]
    latest_high = high.iloc[-1]
    latest_low = low.iloc[-1]
    latest_volume = volume.iloc[-1]

    latest_ma5 = df["MA5"].iloc[-1]
    latest_ma10 = df["MA10"].iloc[-1]
    latest_ma25 = df["MA25"].iloc[-1]
    latest_ma75 = df["MA75"].iloc[-1]
    latest_ma200 = df["MA200"].iloc[-1]

    if pd.isna(latest_ma25) or pd.isna(latest_ma75) or latest_close <= 0:
        return None

    if use_ma200_filter and pd.notna(latest_ma200) and latest_close < latest_ma200:
        return None

    ret_3d = (latest_close / close.iloc[-4] - 1) * 100 if len(close) >= 4 and close.iloc[-4] > 0 else np.nan
    ret_5d = (latest_close / close.iloc[-6] - 1) * 100 if len(close) >= 6 and close.iloc[-6] > 0 else np.nan
    ret_20d = (latest_close / close.iloc[-21] - 1) * 100 if len(close) >= 21 and close.iloc[-21] > 0 else np.nan

    high20 = high.tail(20).max()
    high60 = high.tail(60).max()
    high20_distance_pct = (high20 / latest_close - 1) * 100 if latest_close > 0 else np.nan
    high60_distance_pct = (high60 / latest_close - 1) * 100 if latest_close > 0 else np.nan

    volume_avg5 = volume.iloc[-6:-1].mean() if len(volume) >= 6 else np.nan
    volume_avg20 = volume.tail(20).mean() if len(volume) >= 20 else np.nan
    volume_ratio = latest_volume / volume_avg5 if pd.notna(volume_avg5) and volume_avg5 > 0 else np.nan
    volume_ratio20 = latest_volume / volume_avg20 if pd.notna(volume_avg20) and volume_avg20 > 0 else np.nan

    ma25_deviation_pct = (latest_close / latest_ma25 - 1) * 100 if latest_ma25 > 0 else np.nan
    ma75_deviation_pct = (latest_close / latest_ma75 - 1) * 100 if latest_ma75 > 0 else np.nan

    ma25_up = latest_ma25 > df["MA25"].iloc[-6] if pd.notna(df["MA25"].iloc[-6]) else False
    ma75_up = latest_ma75 > df["MA75"].iloc[-11] if len(df) >= 86 and pd.notna(df["MA75"].iloc[-11]) else False
    above_ma25 = latest_close > latest_ma25
    above_ma75 = latest_close > latest_ma75
    above_ma5 = pd.notna(latest_ma5) and latest_close > latest_ma5
    above_ma10 = pd.notna(latest_ma10) and latest_close > latest_ma10

    close_position = (latest_close - latest_low) / (latest_high - latest_low) if latest_high > latest_low else 0.5
    upper_wick_pct = (latest_high - latest_close) / (latest_high - latest_low) * 100 if latest_high > latest_low else 0
    is_bullish = latest_close > latest_open

    # スコア化。満点100点。
    score = 0
    score_detail = []

    if above_ma25:
        score += 12
        score_detail.append("25日線上")
    if above_ma75:
        score += 12
        score_detail.append("75日線上")
    if ma25_up:
        score += 10
        score_detail.append("25日線上向き")
    if ma75_up:
        score += 8
        score_detail.append("75日線上向き")
    if pd.notna(ret_5d) and ret_5d >= params["min_ret_5d"]:
        score += 15
        score_detail.append("5日上昇強い")
    elif pd.notna(ret_5d) and ret_5d >= params["min_ret_5d"] * 0.6:
        score += 8
        score_detail.append("5日上昇あり")
    if pd.notna(ret_20d) and ret_20d >= params["min_ret_20d"]:
        score += 10
        score_detail.append("20日上昇強い")
    if pd.notna(high20_distance_pct) and high20_distance_pct <= params["high20_distance_max"]:
        score += 15
        score_detail.append("20日高値圏")
    elif pd.notna(high20_distance_pct) and high20_distance_pct <= params["high20_distance_max"] * 1.8:
        score += 8
        score_detail.append("20日高値近い")
    if pd.notna(volume_ratio) and volume_ratio >= params["volume_ratio_min"]:
        score += 10
        score_detail.append("出来高増")
    elif pd.notna(volume_ratio) and volume_ratio >= 1.0:
        score += 5
        score_detail.append("出来高普通")
    if close_position >= params["close_position_min"]:
        score += 8
        score_detail.append("終値位置強い")

    score = min(score, 100)

    # モメンタムのタイプ分け
    if pd.notna(high20_distance_pct) and high20_distance_pct <= 0.5 and pd.notna(volume_ratio) and volume_ratio >= params["volume_ratio_min"]:
        momentum_type = "高値更新初動"
    elif above_ma5 and above_ma10 and pd.notna(ret_5d) and ret_5d >= params["min_ret_5d"]:
        momentum_type = "強勢継続"
    elif pd.notna(ma25_deviation_pct) and ma25_deviation_pct > params["ma25_deviation_max"]:
        momentum_type = "過熱注意"
    else:
        momentum_type = "監視"

    # 売買ライン
    breakout_line = high.tail(5).max()
    recent_low = low.tail(5).min()
    ma10_stop = latest_ma10 * 0.98 if pd.notna(latest_ma10) else np.nan
    ma25_stop = latest_ma25 * 0.97 if pd.notna(latest_ma25) else np.nan

    # モメンタムは損切りを近く置きたい。直近安値と10日線下を優先。
    stop_candidates = [x for x in [recent_low, ma10_stop, ma25_stop] if pd.notna(x)]
    stop_line = min(stop_candidates) if stop_candidates else np.nan

    # 第1利確は+10%、第2利確は+20%または60日高値を参考
    take_profit_1 = breakout_line * 1.10 if pd.notna(breakout_line) else np.nan
    take_profit_2 = max(breakout_line * 1.20, high60) if pd.notna(breakout_line) and pd.notna(high60) else np.nan

    risk = breakout_line - stop_line if pd.notna(breakout_line) and pd.notna(stop_line) else np.nan
    reward1 = take_profit_1 - breakout_line if pd.notna(take_profit_1) and pd.notna(breakout_line) else np.nan
    reward2 = take_profit_2 - breakout_line if pd.notna(take_profit_2) and pd.notna(breakout_line) else np.nan
    loss_pct = risk / breakout_line * 100 if pd.notna(risk) and breakout_line > 0 else np.nan
    rr1 = reward1 / risk if pd.notna(reward1) and pd.notna(risk) and risk > 0 else np.nan
    rr2 = reward2 / risk if pd.notna(reward2) and pd.notna(risk) and risk > 0 else np.nan

    danger_signs = []

    if not above_ma25:
        danger_signs.append("25日線下")
    if not above_ma75:
        danger_signs.append("75日線下")
    if not ma25_up:
        danger_signs.append("25日線が上向きでない")
    if pd.notna(ret_5d) and ret_5d < params["min_ret_5d"] * 0.6:
        danger_signs.append("短期上昇不足")
    if pd.notna(volume_ratio) and volume_ratio < 1.0:
        danger_signs.append("出来高不足")
    elif pd.notna(volume_ratio) and volume_ratio < params["volume_ratio_min"]:
        danger_signs.append("出来高やや不足")
    if pd.notna(high20_distance_pct) and high20_distance_pct > params["high20_distance_max"] * 1.8:
        danger_signs.append("20日高値から遠い")
    if pd.notna(ma25_deviation_pct) and ma25_deviation_pct > params["ma25_deviation_max"]:
        danger_signs.append("25日線乖離が大きい")
    if pd.notna(loss_pct) and loss_pct > params["stop_loss_max_pct"]:
        danger_signs.append("損切りが遠い")
    if upper_wick_pct > params["upper_wick_max_pct"]:
        danger_signs.append("上ヒゲが長い")
    if not is_bullish and close_position < params["close_position_min"]:
        danger_signs.append("ローソク足が弱い")
    if pd.notna(rr1) and rr1 < 1.0 and (pd.isna(rr2) or rr2 < 1.5):
        danger_signs.append("RR不足")

    # 候補が多すぎる問題を防ぐため、モメンタムではスコアだけでなく
    # 「必須条件」を追加でチェックする。
    # 特に厳しめ・超厳しめでは、短期上昇・高値圏・出来高・終値位置・損切り幅を必須にする。
    required_failures = []

    if not above_ma25:
        required_failures.append("25日線上ではない")
    if not above_ma75:
        required_failures.append("75日線上ではない")
    if not ma25_up:
        required_failures.append("25日線上向きでない")
    if params.get("require_ma75_up", False) and not ma75_up:
        required_failures.append("75日線上向きでない")
    if params.get("require_bullish", False) and not is_bullish:
        required_failures.append("陽線ではない")
    if pd.isna(ret_3d) or ret_3d < params.get("min_ret_3d", 0.0):
        required_failures.append("3日上昇率不足")
    if pd.isna(ret_5d) or ret_5d < params["min_ret_5d"]:
        required_failures.append("5日上昇率不足")
    if pd.isna(ret_20d) or ret_20d < params["min_ret_20d"]:
        required_failures.append("20日上昇率不足")
    if pd.isna(high20_distance_pct) or high20_distance_pct > params["high20_distance_max"]:
        required_failures.append("20日高値圏ではない")
    if pd.isna(high60_distance_pct) or high60_distance_pct > params.get("high60_distance_max", 999):
        required_failures.append("60日高値圏ではない")
    if pd.isna(volume_ratio) or volume_ratio < params["volume_ratio_min"]:
        required_failures.append("5日平均比の出来高不足")
    if pd.isna(volume_ratio20) or volume_ratio20 < params.get("volume_ratio20_min", 0):
        required_failures.append("20日平均比の出来高不足")
    if close_position < params["close_position_min"]:
        required_failures.append("終値位置が弱い")
    if pd.notna(ma25_deviation_pct) and ma25_deviation_pct > params["ma25_deviation_max"]:
        required_failures.append("25日線から離れすぎ")
    if upper_wick_pct > params["upper_wick_max_pct"]:
        required_failures.append("上ヒゲが長い")
    if pd.isna(loss_pct) or loss_pct > params["stop_loss_max_pct"]:
        required_failures.append("損切りが遠い")

    if params.get("hard_candidate_filter", False):
        # 候補：必須条件を全部クリア
        if score >= params["min_score_candidate"] and len(required_failures) == 0:
            buy_judge = "候補"
            priority = "A"
        # 監視：多少の不足は許すが、最低限の強さは必要
        elif score >= params["min_score_watch"] and len(required_failures) <= params.get("max_watch_failures", 2):
            buy_judge = "監視"
            priority = "B"
        else:
            buy_judge = "見送り"
            priority = "C"
    else:
        if score >= params["min_score_candidate"] and len(danger_signs) <= 1 and pd.notna(loss_pct) and loss_pct <= params["stop_loss_max_pct"]:
            buy_judge = "候補"
            priority = "A"
        elif score >= params["min_score_watch"] and len(danger_signs) <= params.get("max_watch_failures", 3):
            buy_judge = "監視"
            priority = "B"
        else:
            buy_judge = "見送り"
            priority = "C"

    all_reasons = danger_signs.copy()
    if required_failures:
        all_reasons.append("必須条件NG: " + " / ".join(required_failures))

    skip_reason = "なし" if not all_reasons else " / ".join(all_reasons)

    return {
        "コード": code,
        "銘柄名": name,
        "市場": market,
        "33業種": sector33,
        "17業種": sector17,
        "テーマ": theme,
        "買い候補": buy_judge,
        "優先度": priority,
        "モメンタム種類": momentum_type,
        "スコア": int(score),
        "現在値": round(latest_close, 1),
        "3日上昇率%": round(ret_3d, 2) if pd.notna(ret_3d) else np.nan,
        "5日上昇率%": round(ret_5d, 2) if pd.notna(ret_5d) else np.nan,
        "20日上昇率%": round(ret_20d, 2) if pd.notna(ret_20d) else np.nan,
        "20日高値距離%": round(high20_distance_pct, 2) if pd.notna(high20_distance_pct) else np.nan,
        "60日高値距離%": round(high60_distance_pct, 2) if pd.notna(high60_distance_pct) else np.nan,
        "25日線乖離%": round(ma25_deviation_pct, 2) if pd.notna(ma25_deviation_pct) else np.nan,
        "75日線乖離%": round(ma75_deviation_pct, 2) if pd.notna(ma75_deviation_pct) else np.nan,
        "出来高倍率": round(volume_ratio, 2) if pd.notna(volume_ratio) else np.nan,
        "20日出来高倍率": round(volume_ratio20, 2) if pd.notna(volume_ratio20) else np.nan,
        "終値位置": round(close_position, 2),
        "上ヒゲ%": round(upper_wick_pct, 1),
        "25日線上向き": bool(ma25_up),
        "75日線上向き": bool(ma75_up),
        "上抜けライン": round(breakout_line, 1) if pd.notna(breakout_line) else np.nan,
        "損切りライン": round(stop_line, 1) if pd.notna(stop_line) else np.nan,
        "損切り率%": round(loss_pct, 2) if pd.notna(loss_pct) else np.nan,
        "第1利確ライン": round(take_profit_1, 1) if pd.notna(take_profit_1) else np.nan,
        "第2利確ライン": round(take_profit_2, 1) if pd.notna(take_profit_2) else np.nan,
        "第1RR": round(rr1, 2) if pd.notna(rr1) else np.nan,
        "第2RR": round(rr2, 2) if pd.notna(rr2) else np.nan,
        "強い理由": " / ".join(score_detail) if score_detail else "なし",
        "見送り理由": skip_reason,
    }


def show_saved_momentum_results():
    result_key = "screen_result_df_momentum"
    failed_key = "screen_failed_count_momentum"
    market_key = "screen_market_momentum"

    if result_key not in st.session_state:
        return

    result_df = st.session_state[result_key]
    failed_count = st.session_state.get(failed_key, 0)
    saved_market = st.session_state.get(market_key, "")

    st.subheader("抽出結果")

    if result_df is None or result_df.empty:
        st.warning("条件に一致する銘柄は見つかりませんでした。条件を『ゆるめ』にするか、確認銘柄数を増やしてください。")
        return

    st.write(f"抽出銘柄数：{len(result_df)}")
    st.write(f"取得失敗数：{failed_count}")

    col1, col2, col3 = st.columns(3)

    with col1:
        judge_filter = st.selectbox(
            "表示する買い候補",
            ["すべて", "候補のみ", "候補・監視", "監視のみ", "見送りのみ"],
            index=1,
            key="judge_filter_momentum"
        )

    with col2:
        type_filter = st.multiselect(
            "モメンタム種類",
            ["高値更新初動", "強勢継続", "監視", "過熱注意"],
            default=["高値更新初動", "強勢継続"],
            key="type_filter_momentum"
        )

    with col3:
        theme_keyword = st.text_input(
            "テーマ絞り込み",
            value="",
            placeholder="例：半導体、宇宙、防衛、電線、AI",
            key="theme_keyword_momentum"
        )

    display_df = result_df.copy()

    if judge_filter == "候補のみ":
        display_df = display_df[display_df["買い候補"] == "候補"]
    elif judge_filter == "候補・監視":
        display_df = display_df[display_df["買い候補"].isin(["候補", "監視"])]
    elif judge_filter == "監視のみ":
        display_df = display_df[display_df["買い候補"] == "監視"]
    elif judge_filter == "見送りのみ":
        display_df = display_df[display_df["買い候補"] == "見送り"]

    if type_filter:
        display_df = display_df[display_df["モメンタム種類"].isin(type_filter)]

    if theme_keyword.strip():
        key = theme_keyword.strip()
        display_df = display_df[
            display_df["テーマ"].astype(str).str.contains(key, case=False, na=False)
            | display_df["33業種"].astype(str).str.contains(key, case=False, na=False)
            | display_df["17業種"].astype(str).str.contains(key, case=False, na=False)
            | display_df["銘柄名"].astype(str).str.contains(key, case=False, na=False)
        ]

    display_df = display_df.drop(columns=["_買い候補順", "_優先度順"], errors="ignore")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = display_df.to_csv(index=False, encoding="utf-8-sig")
    market_for_filename = str(saved_market).replace("全市場", "all_markets")

    st.download_button(
        label="CSVダウンロード",
        data=csv,
        file_name=f"momentum_{market_for_filename}_trade_plan.csv",
        mime="text/csv",
        key=f"download_momentum_{market_for_filename}"
    )


def run_momentum_screener(
    market_choice,
    preset_name,
    max_scan,
    wait_sec,
    use_ma200_filter
):
    params = MOMENTUM_PRESETS[preset_name]

    try:
        all_stocks = load_jpx_list()
    except Exception as e:
        st.error(f"JPXの銘柄リスト取得に失敗しました: {e}")
        st.stop()

    target_stocks = filter_market(all_stocks, market_choice)

    if target_stocks.empty:
        st.warning("対象銘柄が見つかりませんでした。")
        st.stop()

    total_all = len(target_stocks)

    if max_scan > 0:
        target_stocks = target_stocks.head(max_scan)

    total = len(target_stocks)

    st.write(f"対象市場：{market_choice}")
    st.write(f"対象銘柄数：{total} / 選択市場全体 {total_all}")

    progress_bar = st.progress(0)
    status_area = st.empty()

    results = []
    failed_count = 0

    for n, (_, row) in enumerate(target_stocks.iterrows(), start=1):
        code = row["コード"]
        name = row["銘柄名"]
        market = row["市場"]
        sector33 = row.get("33業種", "")
        sector17 = row.get("17業種", "")
        theme = row.get("テーマ", infer_theme(code, name, sector33, sector17))

        status_area.write(f"{n}/{total} 確認中：{code} {name}")

        try:
            price_df = download_price(code)

            result = analyze_momentum(
                price_df=price_df,
                code=code,
                name=name,
                market=market,
                sector33=sector33,
                sector17=sector17,
                theme=theme,
                params=params,
                use_ma200_filter=use_ma200_filter
            )

            if result is not None:
                results.append(result)

        except Exception:
            failed_count += 1

        progress_bar.progress(n / total)

        if wait_sec > 0:
            time.sleep(wait_sec)

    status_area.write("スクリーニング完了")

    result_key = "screen_result_df_momentum"
    failed_key = "screen_failed_count_momentum"
    market_key = "screen_market_momentum"

    if len(results) == 0:
        st.session_state[result_key] = pd.DataFrame()
        st.session_state[failed_key] = failed_count
        st.session_state[market_key] = market_choice
    else:
        result_df = pd.DataFrame(results)

        buy_rank = {"候補": 0, "監視": 1, "見送り": 2}
        priority_rank = {"A": 0, "B": 1, "C": 2}
        result_df["_買い候補順"] = result_df["買い候補"].map(buy_rank).fillna(9)
        result_df["_優先度順"] = result_df["優先度"].map(priority_rank).fillna(9)

        sort_cols = ["_買い候補順", "_優先度順", "スコア", "5日上昇率%", "出来高倍率"]
        result_df = result_df.sort_values(
            by=sort_cols,
            ascending=[True, True, False, False, False]
        ).reset_index(drop=True)

        st.session_state[result_key] = result_df
        st.session_state[failed_key] = failed_count
        st.session_state[market_key] = market_choice

    show_saved_momentum_results()


def render_momentum_tab(key_prefix):
    st.subheader("モメンタム初動・強勢継続スクリーニング")

    st.write(
        "強い銘柄に素直についていく相場用です。"
        "高値更新・出来高増・短期上昇・移動平均線上の強さをスコア化し、テーマも表示します。"
    )

    col1, col2 = st.columns(2)

    with col1:
        preset_name = st.selectbox(
            "判定の厳しさ",
            ["普通", "厳しめ", "超厳しめ", "ゆるめ"],
            index=1,
            key=f"{key_prefix}_preset"
        )

    with col2:
        params = MOMENTUM_PRESETS[preset_name]
        st.write("現在の条件")
        st.write(f"候補スコア：{params['min_score_candidate']}点以上")
        st.write(f"5日上昇率：{params['min_ret_5d']}%以上目安")
        st.write(f"20日高値距離：{params['high20_distance_max']}%以内目安")
        st.write(f"60日高値距離：{params.get('high60_distance_max', '指定なし')}%以内目安")
        st.write(f"出来高倍率：直近5日平均の{params['volume_ratio_min']}倍以上目安")
        st.write(f"20日平均出来高倍率：{params.get('volume_ratio20_min', 0)}倍以上目安")
        st.write(f"終値位置：{params['close_position_min'] * 100:.0f}%以上目安")
        st.write(f"25日線乖離の過熱目安：+{params['ma25_deviation_max']}%超")
        st.write(f"損切り許容幅：{params['stop_loss_max_pct']}%以内")

    run_button = st.button(
        "モメンタムスクリーニング実行",
        type="primary",
        key=f"{key_prefix}_run"
    )

    if run_button:
        run_momentum_screener(
            market_choice=market_choice,
            preset_name=preset_name,
            max_scan=max_scan,
            wait_sec=wait_sec,
            use_ma200_filter=use_ma200_filter
        )
    else:
        show_saved_momentum_results()


def show_saved_cup_results():
    """
    カップウィズハンドル用の保存結果表示。
    """
    result_key = "screen_result_df_cup"
    failed_key = "screen_failed_count_cup"
    market_key = "screen_market_cup"

    if result_key not in st.session_state:
        return

    result_df = st.session_state[result_key]
    failed_count = st.session_state.get(failed_key, 0)
    saved_market = st.session_state.get(market_key, "")

    st.subheader("抽出結果")

    if result_df is None or result_df.empty:
        st.warning("条件に一致する銘柄は見つかりませんでした。条件を「ゆるめ」にするか、確認銘柄数を増やしてください。")
        return

    st.write(f"抽出銘柄数：{len(result_df)}")
    st.write(f"取得失敗数：{failed_count}")

    judge_filter = st.selectbox(
        "表示する買い候補",
        ["すべて", "候補のみ", "候補・監視", "監視のみ", "見送りのみ"],
        key="judge_filter_cup"
    )

    display_df = result_df.copy()

    if judge_filter == "候補のみ":
        display_df = display_df[display_df["買い候補"] == "候補"]
    elif judge_filter == "候補・監視":
        display_df = display_df[display_df["買い候補"].isin(["候補", "監視"])]
    elif judge_filter == "監視のみ":
        display_df = display_df[display_df["買い候補"] == "監視"]
    elif judge_filter == "見送りのみ":
        display_df = display_df[display_df["買い候補"] == "見送り"]

    timeframe_filter = st.multiselect(
        "表示する足種",
        ["日足", "週足", "月足"],
        default=["日足", "週足", "月足"],
        key="timeframe_filter_cup"
    )

    if timeframe_filter:
        display_df = display_df[display_df["足種"].isin(timeframe_filter)]

    display_df = display_df.drop(columns=["_買い候補順", "_優先度順"], errors="ignore")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = display_df.to_csv(index=False, encoding="utf-8-sig")
    market_for_filename = str(saved_market).replace("全市場", "all_markets")

    st.download_button(
        label="CSVダウンロード",
        data=csv,
        file_name=f"cup_with_handle_{market_for_filename}_trade_plan.csv",
        mime="text/csv",
        key=f"download_cup_{market_for_filename}"
    )


def run_cup_screener(
    market_choice,
    preset_name,
    target_timeframes,
    max_scan,
    wait_sec,
    use_ma200_filter
):
    params = CUP_PRESETS[preset_name]

    try:
        all_stocks = load_jpx_list()
    except Exception as e:
        st.error(f"JPXの銘柄リスト取得に失敗しました: {e}")
        st.stop()

    target_stocks = filter_market(all_stocks, market_choice)

    if target_stocks.empty:
        st.warning("対象銘柄が見つかりませんでした。")
        st.stop()

    total_all = len(target_stocks)

    if max_scan > 0:
        target_stocks = target_stocks.head(max_scan)

    total = len(target_stocks)

    st.write(f"対象市場：{market_choice}")
    st.write(f"対象銘柄数：{total} / 選択市場全体 {total_all}")
    st.write(f"対象足種：{', '.join(target_timeframes)}")

    progress_bar = st.progress(0)
    status_area = st.empty()

    results = []
    failed_count = 0

    for n, (_, row) in enumerate(target_stocks.iterrows(), start=1):
        code = row["コード"]
        name = row["銘柄名"]
        market = row["市場"]

        status_area.write(f"{n}/{total} 確認中：{code} {name}")

        try:
            price_df = download_price_long(code)

            detected_list = analyze_cup_with_handle(
                price_df=price_df,
                code=code,
                name=name,
                market=market,
                params=params,
                target_timeframes=target_timeframes,
                use_ma200_filter=use_ma200_filter
            )

            if detected_list:
                results.extend(detected_list)

        except Exception:
            failed_count += 1

        progress_bar.progress(n / total)

        if wait_sec > 0:
            time.sleep(wait_sec)

    status_area.write("スクリーニング完了")

    result_key = "screen_result_df_cup"
    failed_key = "screen_failed_count_cup"
    market_key = "screen_market_cup"

    if len(results) == 0:
        st.session_state[result_key] = pd.DataFrame()
        st.session_state[failed_key] = failed_count
        st.session_state[market_key] = market_choice
    else:
        result_df = pd.DataFrame(results)

        buy_rank = {"候補": 0, "監視": 1, "見送り": 2}
        priority_rank = {"A": 0, "B": 1, "C": 2}
        timeframe_rank = {"日足": 0, "週足": 1, "月足": 2}
        result_df["_買い候補順"] = result_df["買い候補"].map(buy_rank).fillna(9)
        result_df["_優先度順"] = result_df["優先度"].map(priority_rank).fillna(9)
        result_df["_足種順"] = result_df["足種"].map(timeframe_rank).fillna(9)

        sort_cols = ["_買い候補順", "_優先度順", "_足種順", "第1RR", "第2RR", "スコア"]
        result_df = result_df.sort_values(
            by=sort_cols,
            ascending=[True, True, True, False, False, False]
        ).reset_index(drop=True)

        st.session_state[result_key] = result_df
        st.session_state[failed_key] = failed_count
        st.session_state[market_key] = market_choice

    show_saved_cup_results()


def render_cup_tab(key_prefix):
    st.subheader("カップウィズハンドル上抜け接近スクリーニング")

    st.write(
        "日足・週足・月足それぞれで、カップウィズハンドルを形成し、"
        "ピボット上抜けが近い銘柄、または出来高増で上抜け始めた銘柄を探します。"
    )

    col1, col2 = st.columns(2)

    with col1:
        preset_name = st.selectbox(
            "判定の厳しさ",
            ["普通", "厳しめ", "超厳しめ", "ゆるめ"],
            index=1,
            key=f"{key_prefix}_preset"
        )

        target_timeframes = st.multiselect(
            "確認する足種",
            ["日足", "週足", "月足"],
            default=["日足", "週足", "月足"],
            key=f"{key_prefix}_timeframes"
        )

    with col2:
        params = CUP_PRESETS[preset_name]
        st.write("現在の条件")
        st.write(f"ピボットまでの距離：{params['pivot_distance_max_pct']}%以内")
        st.write(f"カップ深さ：{params['min_cup_depth_pct']}% 〜 {params['max_cup_depth_pct']}%")
        st.write(f"ハンドル深さ：{params['min_handle_depth_pct']}% 〜 {params['max_handle_depth_pct']}%")
        st.write(f"左右リム差：{params['rim_tolerance_pct']}%以内")

    if not target_timeframes:
        st.warning("確認する足種を1つ以上選んでください。")
        show_saved_cup_results()
        return

    run_button = st.button(
        "カップウィズハンドルをスクリーニング実行",
        type="primary",
        key=f"{key_prefix}_run"
    )

    if run_button:
        run_cup_screener(
            market_choice=market_choice,
            preset_name=preset_name,
            target_timeframes=target_timeframes,
            max_scan=max_scan,
            wait_sec=wait_sec,
            use_ma200_filter=use_ma200_filter
        )
    else:
        show_saved_cup_results()


# =========================
# 適時開示PDF材料判定タブ
# =========================
JST = timezone(timedelta(hours=9))
YANOSHIN_BASE_URL = "https://webapi.yanoshin.jp/webapi/tdnet/list"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StreamlitTDnetMaterialScreener/1.0)"
}


POSITIVE_DISCLOSURE_RULES = [
    (r"公開買付け|TOB|応募推奨|賛同の意見表明", 90, "TOB/公開買付け関連"),
    (r"上方修正|上方に修正|業績予想.*増額|予想.*上回", 45, "業績上方修正"),
    (r"増配|復配|配当予想.*増額|配当.*引き上げ", 35, "増配/復配"),
    (r"黒字転換|営業黒字|経常黒字", 45, "黒字転換"),
    (r"自己株式の取得(?!状況)|自己株式取得に係る事項の決定|ToSTNeT-3.*買付|自己株式.*消却", 28, "自社株買い/消却"),
    (r"資本業務提携|業務提携|戦略的業務提携|共同開発|事業協力", 28, "提携/共同開発"),
    (r"大型受注|受注|契約締結|基本契約|販売契約|採用|導入決定", 30, "受注/契約/採用"),
    (r"中期経営計画|新中期経営計画|成長戦略|上場維持基準.*適合", 15, "中計/成長戦略"),
    (r"株式分割", 22, "株式分割"),
    (r"特別利益|補助金|助成金", 18, "特別利益/補助金"),
]

NEGATIVE_DISCLOSURE_RULES = [
    (r"監査意見不表明|意見不表明|限定付適正意見|監査法人.*辞任", 90, "監査意見リスク"),
    (r"不適切会計|第三者委員会|調査報告書|内部統制.*不備", 75, "不適切会計/第三者委員会"),
    (r"下方修正|下方に修正|業績予想.*減額|予想.*下回", 50, "業績下方修正"),
    (r"赤字転落|営業赤字|経常赤字|債務超過|継続企業の前提", 60, "赤字/継続企業リスク"),
    (r"減配|無配|配当予想.*減額|配当.*引き下げ", 45, "減配/無配"),
    (r"MSワラント|行使価額修正条項付|行使価額修正型|新株予約権.*発行|第三者割当.*新株|希薄化|転換社債型新株予約権付社債", 55, "希薄化/ワラント"),
    (r"公募増資|株式の売出し|新株式発行", 45, "増資/売出し"),
    (r"減損損失|特別損失|評価損|貸倒引当金", 32, "減損/特別損失"),
    (r"決算発表.*延期|提出遅延|有価証券報告書.*延長", 35, "決算/提出遅延"),
    (r"訴訟|行政処分|業務停止|不正アクセス|情報漏えい", 35, "訴訟/行政処分/事故"),
]

LOW_IMPORTANCE_PATTERNS = [
    r"自己株式の取得状況|自己株式取得状況|月間行使状況|払込完了|定款の一部変更|人事異動|組織変更|役員.*異動|株主総会",
    r"決算説明資料|説明会資料|書き起こし|コーポレート・ガバナンス|独立役員|支配株主",
]

MIXED_PATTERNS = [
    r"上方修正.*減配|増配.*下方修正|特別利益.*下方修正|提携.*新株予約権|業務提携.*第三者割当",
]

DISCLOSURE_THEME_KEYWORDS = [
    ("AI/データセンター", ["AI", "人工知能", "データセンター", "GPU", "生成AI", "LLM", "半導体", "サーバー"]),
    ("宇宙", ["宇宙", "衛星", "ロケット", "スペース", "軌道", "SAR"]),
    ("防衛", ["防衛", "自衛隊", "安全保障", "ミサイル", "ドローン", "無人機"]),
    ("電力/原発", ["電力", "原子力", "原発", "送電", "変電", "蓄電", "電池"]),
    ("半導体/電子部品", ["半導体", "電子部品", "MLCC", "パワー半導体", "SiC", "GaN", "センサー"]),
    ("HDD/ストレージ", ["HDD", "ハードディスク", "ストレージ", "磁気ヘッド", "プラッタ", "データストレージ"]),
    ("バイオ/医薬", ["治験", "承認", "臨床", "医薬", "バイオ", "細胞", "製剤", "FDA"]),
    ("M&A/TOB", ["TOB", "公開買付", "買収", "子会社化", "株式取得", "譲渡"]),
    ("株主還元", ["増配", "自己株式", "株式分割", "配当", "消却"]),
    ("暗号資産/金", ["ビットコイン", "暗号資産", "Bitcoin", "ゴールド", "金"]),
]


def normalize_disclosure_code(code):
    if code is None:
        return ""
    s = str(code).strip().upper()
    s = re.sub(r"[^0-9A-Z]", "", s)
    if len(s) >= 5 and s.endswith("0"):
        s = s[:-1]
    return s


def direct_pdf_url(url):
    if not url:
        return ""
    url = str(url).strip()
    if ".pdf" in url.lower() and "rd.php" not in url:
        return url
    if "rd.php?" in url:
        encoded = url.split("rd.php?", 1)[1]
        decoded = unquote(encoded).strip().rstrip("=")
        if ".pdf" in decoded.lower():
            return decoded
    return url


def parse_tdnet_datetime(value):
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"]:
        try:
            return datetime.strptime(text[:19], fmt)
        except Exception:
            pass
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\s+(\d{1,2}):(\d{2})", text)
    if m:
        y, mo, d, h, mi = map(int, m.groups())
        return datetime(y, mo, d, h, mi)
    return None


@st.cache_data(ttl=60)
def fetch_tdnet_disclosures_free(target_date, max_items=200):
    """
    無料で使える非公式API/HTMLから適時開示一覧を取得する。
    非公式のため、取得できない場合に備えて手動PDFアップロードも併設している。
    """
    ymd = target_date.strftime("%Y%m%d")
    candidates = [
        f"{YANOSHIN_BASE_URL}/{ymd}.json?limit={max_items}",
        f"{YANOSHIN_BASE_URL}/{ymd}.json2?limit={max_items}",
        f"{YANOSHIN_BASE_URL}/recent.json?limit={max_items}",
        f"{YANOSHIN_BASE_URL}/recent.json2?limit={max_items}",
    ]

    rows = []
    last_error = ""
    for url in candidates:
        try:
            res = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
            if res.status_code != 200 or not res.text.strip():
                last_error = f"{url}: status {res.status_code}"
                continue
            data = res.json()
            items = data.get("items", data if isinstance(data, list) else [])
            for item in items:
                td = item.get("Tdnet", item) if isinstance(item, dict) else {}
                title = td.get("title") or td.get("Title") or td.get("document_title") or td.get("name") or ""
                company = td.get("company_name") or td.get("company") or td.get("CompanyName") or td.get("companyname") or ""
                code = td.get("code") or td.get("Code") or td.get("sec_code") or td.get("company_code") or ""
                pubdate = td.get("pubdate") or td.get("published_at") or td.get("datetime") or td.get("date") or ""
                pdf_url = (
                    td.get("url") or td.get("Url") or td.get("link") or td.get("pdf_url") or
                    td.get("document_url") or td.get("tdnet_url") or ""
                )
                dt = parse_tdnet_datetime(pubdate)
                if not title:
                    continue
                if dt and dt.date() != target_date:
                    continue
                rows.append({
                    "開示日時": dt,
                    "時刻": dt.strftime("%H:%M") if dt else "",
                    "コード": normalize_disclosure_code(code),
                    "銘柄名": str(company).strip(),
                    "タイトル": str(title).strip(),
                    "PDFリンク": direct_pdf_url(pdf_url),
                    "取得元": "Yanoshin JSON",
                })
            if rows:
                break
        except Exception as e:
            last_error = f"{url}: {e}"
            continue

    if not rows:
        html_rows = fetch_tdnet_disclosures_from_html(target_date, max_items=max_items)
        rows.extend(html_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, last_error
    df = df.drop_duplicates(subset=["コード", "タイトル", "PDFリンク"])
    if "開示日時" in df.columns:
        df = df.sort_values("開示日時", ascending=False, na_position="last")
    return df.head(max_items).reset_index(drop=True), last_error


@st.cache_data(ttl=60)
def fetch_tdnet_disclosures_from_html(target_date, max_items=200):
    if BeautifulSoup is None:
        return []
    ymd = target_date.strftime("%Y%m%d")
    candidates = [
        f"{YANOSHIN_BASE_URL}/{ymd}.html",
        f"{YANOSHIN_BASE_URL}/recent.html",
    ]
    rows = []
    for url in candidates:
        try:
            res = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            for a in soup.find_all("a"):
                href = a.get("href", "")
                if ".pdf" not in href.lower() and "rd.php" not in href.lower():
                    continue
                pdf_url = direct_pdf_url(urljoin(url, href))
                title = a.get_text(" ", strip=True)
                if not title:
                    continue
                container = a.find_parent(["tr", "li", "p", "div"]) or a.parent
                text = container.get_text(" ", strip=True) if container else title
                dt = parse_tdnet_datetime(text)
                if dt and dt.date() != target_date:
                    continue
                code = ""
                company = ""
                m = re.search(r"\[\s*([0-9A-Z]{4,5})\s*\]\s*([^\s　]+)?", text)
                if m:
                    code = normalize_disclosure_code(m.group(1))
                    company = (m.group(2) or "").strip()
                rows.append({
                    "開示日時": dt,
                    "時刻": dt.strftime("%H:%M") if dt else "",
                    "コード": code,
                    "銘柄名": company,
                    "タイトル": title,
                    "PDFリンク": pdf_url,
                    "取得元": "Yanoshin HTML",
                })
            if rows:
                break
        except Exception:
            continue
    return rows[:max_items]


@st.cache_data(ttl=60 * 15)
def extract_pdf_text_from_url(pdf_url, max_pages=6):
    if not pdf_url:
        return "", "PDFリンクなし"
    if fitz is None:
        return "", "PyMuPDFが未インストールです。requirements.txtに PyMuPDF を追加してください。"
    try:
        res = requests.get(pdf_url, headers=REQUEST_HEADERS, timeout=25)
        if res.status_code != 200:
            return "", f"PDF取得失敗 status={res.status_code}"
        doc = fitz.open(stream=res.content, filetype="pdf")
        parts = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            parts.append(page.get_text("text"))
        text = "\n".join(parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text, ""
    except Exception as e:
        return "", str(e)


def extract_pdf_text_from_bytes(file_bytes, max_pages=6):
    if fitz is None:
        return "", "PyMuPDFが未インストールです。requirements.txtに PyMuPDF を追加してください。"
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        parts = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            parts.append(page.get_text("text"))
        text = "\n".join(parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text, ""
    except Exception as e:
        return "", str(e)


def match_rules(text, rules):
    hits = []
    score = 0
    for pattern, points, label in rules:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(label)
            score += points
    return score, list(dict.fromkeys(hits))


def extract_material_numbers(text):
    if not text:
        return ""
    key_pattern = r"(売上高|営業利益|経常利益|純利益|当期純利益|EBITDA|配当|自己株式|取得価額|希薄化|発行済株式|受注|契約|特別利益|特別損失|減損|上方修正|下方修正)"
    sentences = re.split(r"(?<=[。．])|\n", text)
    snippets = []
    for s in sentences:
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) < 8:
            continue
        if re.search(key_pattern, s) and re.search(r"[0-9０-９]+|億円|百万円|%|％|円", s):
            snippets.append(s[:120])
        if len(snippets) >= 4:
            break
    return " / ".join(snippets)


def infer_disclosure_theme(code, company, title, body_text):
    base = infer_theme(normalize_disclosure_code(code), company, "", "")
    joined = f"{title}\n{body_text[:1200]}"
    hits = []
    for theme, keywords in DISCLOSURE_THEME_KEYWORDS:
        if any(k.lower() in joined.lower() for k in keywords):
            hits.append(theme)
    if base and base != "個別テーマ要確認":
        hits.insert(0, base)
    hits = list(dict.fromkeys(hits))
    return " / ".join(hits[:3]) if hits else "個別テーマ要確認"


def judge_disclosure_material(title, body_text, code="", company=""):
    joined = f"{title}\n{body_text}".strip()
    compact = re.sub(r"\s+", " ", joined)

    pos_score, pos_hits = match_rules(compact, POSITIVE_DISCLOSURE_RULES)
    neg_score, neg_hits = match_rules(compact, NEGATIVE_DISCLOSURE_RULES)

    low_importance = any(re.search(p, compact, flags=re.IGNORECASE) for p in LOW_IMPORTANCE_PATTERNS)
    mixed = any(re.search(p, compact, flags=re.IGNORECASE) for p in MIXED_PATTERNS)

    # 月次の自己株取得状況などは、好材料扱いしすぎないように補正
    if re.search(r"自己株式の取得状況|月間行使状況", title):
        pos_score = min(pos_score, 8)
        if "行使価額修正" in compact or "新株予約権" in compact:
            neg_score = max(neg_score, 18)

    total = pos_score - neg_score

    if mixed and pos_score >= 25 and neg_score >= 25:
        material = "好悪混在"
        importance = "A" if max(pos_score, neg_score) >= 55 else "B"
        next_action = "翌日監視は可。ただし希薄化・減配など悪材料部分を必ず確認"
    elif total >= 75:
        material = "好材料"
        importance = "S"
        next_action = "翌日最優先で監視。寄り前気配と板の強さを確認"
    elif total >= 40:
        material = "好材料"
        importance = "A"
        next_action = "翌日監視候補。寄り付き位置と出来高を確認"
    elif total >= 18:
        material = "好材料"
        importance = "B"
        next_action = "監視候補。単独材料としてはやや弱め"
    elif total <= -75:
        material = "悪材料"
        importance = "S"
        next_action = "買い対象外。保有ならリスク確認優先"
    elif total <= -40:
        material = "悪材料"
        importance = "A"
        next_action = "基本見送り。寄り後のリバ狙いも慎重"
    elif total <= -18:
        material = "悪材料"
        importance = "B"
        next_action = "見送り寄り。内容確認のみ"
    else:
        material = "要確認" if ("修正" in title or "決算" in title or "差異" in title) else "中立"
        importance = "C"
        next_action = "単独では売買材料にしにくい。本文と翌日反応を確認"

    if low_importance and abs(total) < 30:
        material = "中立"
        importance = "C"
        next_action = "重要度低め。基本は売買材料にしない"

    numbers = extract_material_numbers(compact)
    theme = infer_disclosure_theme(code, company, title, body_text)

    return {
        "材料判定": material,
        "重要度": importance,
        "材料スコア": int(total),
        "好材料ポイント": " / ".join(pos_hits) if pos_hits else "",
        "悪材料ポイント": " / ".join(neg_hits) if neg_hits else "",
        "本文から拾った数字": numbers,
        "テーマ": theme,
        "翌日判断": next_action,
    }


def filter_disclosures_by_time(df, after_time_text):
    if df.empty or "開示日時" not in df.columns:
        return df
    hour, minute = map(int, after_time_text.split(":"))
    return df[df["開示日時"].apply(lambda x: False if pd.isna(x) else (x.hour, x.minute) >= (hour, minute))].copy()


def render_disclosure_result_table(result_df):
    if result_df.empty:
        st.warning("条件に合う適時開示がありませんでした。日付・時刻・件数を変えて試してください。")
        return

    order = ["時刻", "コード", "銘柄名", "タイトル", "材料判定", "重要度", "材料スコア", "テーマ", "好材料ポイント", "悪材料ポイント", "本文から拾った数字", "翌日判断", "PDFリンク", "PDF読取エラー"]
    cols = [c for c in order if c in result_df.columns]
    view_df = result_df[cols].copy()
    st.dataframe(
        view_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PDFリンク": st.column_config.LinkColumn("PDFリンク")
        } if "PDFリンク" in view_df.columns else None,
    )

    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "結果をCSVでダウンロード",
        data=csv,
        file_name="tdnet_pdf_material_judgement.csv",
        mime="text/csv",
    )

    st.subheader("詳細確認")
    choices = (result_df["コード"].astype(str) + " " + result_df["銘柄名"].astype(str) + "｜" + result_df["タイトル"].astype(str)).tolist()
    selected = st.selectbox("詳細を見る開示", choices, key="tdnet_detail_select")
    idx = choices.index(selected)
    row = result_df.iloc[idx]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("材料判定", row.get("材料判定", "-"))
    col2.metric("重要度", row.get("重要度", "-"))
    col3.metric("材料スコア", row.get("材料スコア", "-"))
    col4.metric("テーマ", row.get("テーマ", "-"))
    st.write("### 判断ポイント")
    st.write("好材料：", row.get("好材料ポイント", ""))
    st.write("悪材料：", row.get("悪材料ポイント", ""))
    st.write("本文数値：", row.get("本文から拾った数字", ""))
    st.write("翌日判断：", row.get("翌日判断", ""))
    if row.get("PDFリンク", ""):
        st.link_button("PDFを開く", row.get("PDFリンク"))


def render_tdnet_pdf_tab(tab_key="tdnet_pdf"):
    st.subheader("適時開示PDF材料判定")
    st.write("無料で使える範囲として、非公式TDnet一覧からPDFを取得し、PDF本文をルール判定します。場中開示も引け後開示も確認できます。OpenAI APIキーやJ-Quants有料アドオンは不要です。")
    st.warning("無料版はAIではなくルール判定です。PDFのレイアウトや非公式APIの停止・遅延により、取得/判定できない場合があります。重要開示は必ずPDF原文で確認してください。")

    mode = st.radio(
        "使い方",
        ["当日の適時開示を取得して判定", "PDFを手動アップロードして判定"],
        horizontal=True,
        key=f"{tab_key}_mode",
    )

    if mode == "当日の適時開示を取得して判定":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            target_date = st.date_input("開示日", value=datetime.now(JST).date(), key=f"{tab_key}_date")
        with c2:
            after_time_label = st.selectbox(
                "この時刻以降",
                [
                    "00:00 全開示",
                    "09:00 場中含む",
                    "11:30 前引け後",
                    "12:30 後場以降",
                    "15:00 大引け後",
                    "15:30 決算集中時間以降",
                    "16:00 16時以降",
                    "17:00 17時以降",
                ],
                index=1,
                key=f"{tab_key}_after",
            )
            after_time = after_time_label.split()[0]
        with c3:
            max_items = st.number_input(
                "取得する最大件数",
                min_value=20,
                max_value=1200,
                value=600,
                step=50,
                key=f"{tab_key}_max_items",
                help="場中から全部見る場合は多めにします。重い場合は300〜600件に下げてください。",
            )
        with c4:
            max_pdf_count = st.number_input(
                "PDF本文を読む件数",
                min_value=5,
                max_value=500,
                value=150,
                step=25,
                key=f"{tab_key}_max_pdf",
                help="件数を増やすほど時間がかかります。Streamlit Cloudで止まる場合は100〜150件程度にしてください。",
            )

        c5, c6, c7 = st.columns(3)
        with c5:
            max_pages = st.number_input("1PDFあたり読むページ数", min_value=1, max_value=30, value=8, step=1, key=f"{tab_key}_pages")
        with c6:
            title_keyword = st.text_input("タイトル絞り込み 任意", value="", key=f"{tab_key}_kw")
        with c7:
            only_material = st.checkbox("S/A/B材料だけ表示", value=True, key=f"{tab_key}_only_material")

        c8, c9 = st.columns(2)
        with c8:
            only_with_pdf = st.checkbox("PDFリンクありだけ読む", value=True, key=f"{tab_key}_only_pdf")
        with c9:
            st.caption("場中開示も読む場合は、時刻を『09:00 場中含む』か『00:00 全開示』にしてください。")

        if st.button("適時開示PDFを判定", type="primary", key=f"{tab_key}_run"):
            source_df, error_msg = fetch_tdnet_disclosures_free(target_date, max_items=int(max_items))
            if source_df.empty:
                st.error("適時開示一覧を取得できませんでした。非公式API側の停止/遅延の可能性があります。手動PDFアップロードを使ってください。")
                if error_msg:
                    st.caption(error_msg)
                return

            source_df = filter_disclosures_by_time(source_df, after_time)
            if title_keyword.strip():
                kw = title_keyword.strip()
                source_df = source_df[source_df["タイトル"].astype(str).str.contains(kw, case=False, na=False)]

            if only_with_pdf and "PDFリンク" in source_df.columns:
                source_df = source_df[source_df["PDFリンク"].astype(str).str.len() > 0]

            st.write(f"取得件数：{len(source_df)}件。PDF本文読取：最大{int(max_pdf_count)}件")
            if int(max_pdf_count) >= 200:
                st.info("PDF読取件数が多いため、完了まで時間がかかります。途中で止まる場合は件数を150件前後に下げてください。")
            rows = []
            progress = st.progress(0)
            status = st.empty()
            work_df = source_df.head(int(max_pdf_count)).reset_index(drop=True)
            total = len(work_df)
            for i, row in work_df.iterrows():
                status.write(f"PDF読取中：{row.get('コード','')} {row.get('銘柄名','')} {i + 1}/{total}")
                pdf_text, pdf_error = extract_pdf_text_from_url(row.get("PDFリンク", ""), max_pages=int(max_pages))
                judgement = judge_disclosure_material(
                    title=row.get("タイトル", ""),
                    body_text=pdf_text,
                    code=row.get("コード", ""),
                    company=row.get("銘柄名", ""),
                )
                out = row.to_dict()
                out.update(judgement)
                out["PDF読取エラー"] = pdf_error
                rows.append(out)
                progress.progress((i + 1) / total if total else 1.0)
            status.empty()

            result_df = pd.DataFrame(rows)
            if not result_df.empty:
                importance_order = {"S": 4, "A": 3, "B": 2, "C": 1}
                result_df["重要度順"] = result_df["重要度"].map(importance_order).fillna(0)
                result_df = result_df.sort_values(["重要度順", "材料スコア", "開示日時"], ascending=[False, False, False])
                if only_material:
                    result_df = result_df[result_df["重要度"].isin(["S", "A", "B"])]
                st.session_state[f"{tab_key}_results"] = result_df.reset_index(drop=True)

        if f"{tab_key}_results" in st.session_state:
            render_disclosure_result_table(st.session_state[f"{tab_key}_results"])

    else:
        st.write("TDnet取得が不安定な場合や、特定のPDFだけ判定したい場合に使います。")
        up_files = st.file_uploader("PDFをアップロード", type=["pdf"], accept_multiple_files=True, key=f"{tab_key}_upload")
        c1, c2, c3 = st.columns(3)
        with c1:
            manual_code = st.text_input("コード 任意", key=f"{tab_key}_manual_code")
        with c2:
            manual_company = st.text_input("銘柄名 任意", key=f"{tab_key}_manual_company")
        with c3:
            manual_title = st.text_input("開示タイトル 任意", key=f"{tab_key}_manual_title")
        max_pages = st.number_input("読むページ数", min_value=1, max_value=30, value=8, step=1, key=f"{tab_key}_manual_pages")

        if st.button("アップロードPDFを判定", type="primary", key=f"{tab_key}_manual_run"):
            if not up_files:
                st.error("PDFをアップロードしてください。")
                return
            rows = []
            for f in up_files:
                pdf_text, pdf_error = extract_pdf_text_from_bytes(f.read(), max_pages=int(max_pages))
                title = manual_title.strip() or f.name
                judgement = judge_disclosure_material(title, pdf_text, manual_code, manual_company)
                rows.append({
                    "時刻": "",
                    "コード": normalize_disclosure_code(manual_code),
                    "銘柄名": manual_company,
                    "タイトル": title,
                    "PDFリンク": "",
                    **judgement,
                    "PDF読取エラー": pdf_error,
                })
            result_df = pd.DataFrame(rows)
            st.session_state[f"{tab_key}_manual_results"] = result_df

        if f"{tab_key}_manual_results" in st.session_state:
            render_disclosure_result_table(st.session_state[f"{tab_key}_manual_results"])




# =========================
# 材料 × モメンタム統合スクリーニング
# =========================

def material_importance_points(importance):
    return {"S": 45, "A": 32, "B": 18, "C": 5}.get(str(importance), 0)


def material_judgement_guard(material):
    material = str(material)
    if "悪材料" in material:
        return "悪材料"
    if "好悪混在" in material:
        return "好悪混在"
    if "好材料" in material:
        return "好材料"
    if "要確認" in material:
        return "要確認"
    return "中立"


def calc_material_momentum_rank(judgement, momentum_result):
    """
    PDF材料判定とモメンタム判定を合成する。
    チャート反応というより、現在の地合い・高値圏・出来高の強さと材料の質を合わせて翌日監視優先度を出す。
    """
    material = judgement.get("材料判定", "")
    guard = material_judgement_guard(material)
    importance = judgement.get("重要度", "C")
    material_score = float(judgement.get("材料スコア", 0) or 0)

    if momentum_result is None:
        momentum_score = 0
        buy_judge = "判定不可"
        priority = "C"
        momentum_bonus = 0
    else:
        momentum_score = float(momentum_result.get("スコア", 0) or 0)
        buy_judge = str(momentum_result.get("買い候補", ""))
        priority = str(momentum_result.get("優先度", "C"))
        if buy_judge == "候補":
            momentum_bonus = 24
        elif buy_judge == "監視":
            momentum_bonus = 12
        else:
            momentum_bonus = 0

    score = material_importance_points(importance) + material_score * 1.4 + momentum_score * 0.45 + momentum_bonus

    # 悪材料は買い候補から除外。好悪混在は強くても要確認止まり。
    if guard == "悪材料":
        rank = "見送り"
        action = "買い対象外。悪材料・希薄化・下方修正などの可能性を優先確認"
    elif guard == "好悪混在":
        if score >= 95 and buy_judge in ["候補", "監視"]:
            rank = "A-要確認"
            action = "好材料と悪材料が混在。PTS・翌日寄り前にPDF原文を再確認"
        else:
            rank = "B-要確認"
            action = "好悪混在のため、飛びつかず内容確認優先"
    elif guard == "好材料":
        if score >= 105 and buy_judge == "候補":
            rank = "S"
            action = "最優先監視。好材料＋モメンタム強。寄り付き条件と上抜けラインを確認"
        elif score >= 88 and buy_judge in ["候補", "監視"]:
            rank = "A"
            action = "優先監視。好材料＋チャート高値圏。押し目または上抜けで確認"
        elif score >= 65:
            rank = "B"
            action = "監視候補。材料は良いがモメンタム確認が必要"
        else:
            rank = "C"
            action = "材料単体は確認。チャートが弱ければ様子見"
    elif guard == "要確認":
        if score >= 90 and buy_judge in ["候補", "監視"]:
            rank = "B-要確認"
            action = "タイトルだけでは判断しにくいがモメンタム強。PDF原文確認"
        else:
            rank = "C"
            action = "要確認。数字・継続性を見てから判断"
    else:
        if buy_judge == "候補" and momentum_score >= 90:
            rank = "B"
            action = "材料は中立寄りだがモメンタムは強い。材料名目の物色か確認"
        else:
            rank = "C"
            action = "優先度低め"

    return {
        "総合ランク": rank,
        "総合スコア": round(score, 1),
        "翌日監視判断": action,
    }


def render_material_momentum_result(result_df, key_prefix="matmom"):
    if result_df is None or result_df.empty:
        st.warning("条件に合う材料×モメンタム候補はありませんでした。時刻・件数・判定条件を変えてください。")
        return

    st.subheader("材料 × モメンタム 統合結果")
    st.write(f"表示件数：{len(result_df)}")

    c1, c2, c3 = st.columns(3)
    with c1:
        rank_filter = st.multiselect(
            "表示ランク",
            ["S", "A", "A-要確認", "B", "B-要確認", "C", "見送り"],
            default=["S", "A", "A-要確認", "B", "B-要確認"],
            key=f"{key_prefix}_rank_filter",
        )
    with c2:
        material_filter = st.multiselect(
            "材料判定",
            ["好材料", "好悪混在", "要確認", "中立", "悪材料"],
            default=["好材料", "好悪混在", "要確認"],
            key=f"{key_prefix}_material_filter",
        )
    with c3:
        theme_keyword = st.text_input("テーマ絞り込み", value="", placeholder="例：半導体、AI、防衛、宇宙", key=f"{key_prefix}_theme_kw")

    display_df = result_df.copy()
    if rank_filter:
        display_df = display_df[display_df["総合ランク"].isin(rank_filter)]
    if material_filter:
        display_df = display_df[display_df["材料判定"].isin(material_filter)]
    if theme_keyword.strip():
        kw = theme_keyword.strip()
        display_df = display_df[
            display_df["テーマ"].astype(str).str.contains(kw, case=False, na=False)
            | display_df["銘柄名"].astype(str).str.contains(kw, case=False, na=False)
            | display_df["タイトル"].astype(str).str.contains(kw, case=False, na=False)
        ]

    order = [
        "総合ランク", "総合スコア", "時刻", "コード", "銘柄名", "テーマ", "タイトル",
        "材料判定", "重要度", "材料スコア", "好材料ポイント", "悪材料ポイント", "本文から拾った数字",
        "モメンタム判定", "モメンタム種類", "モメンタムスコア", "5日上昇率%", "20日上昇率%",
        "出来高倍率", "20日高値距離%", "損切り率%", "翌日監視判断", "PDFリンク",
    ]
    cols = [c for c in order if c in display_df.columns]
    st.dataframe(
        display_df[cols],
        use_container_width=True,
        hide_index=True,
        column_config={"PDFリンク": st.column_config.LinkColumn("PDFリンク")} if "PDFリンク" in cols else None,
    )

    csv = display_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "CSVダウンロード",
        data=csv,
        file_name="material_momentum_screening.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )

    if display_df.empty:
        return

    st.subheader("詳細確認")
    choices = (display_df["総合ランク"].astype(str) + "｜" + display_df["コード"].astype(str) + " " + display_df["銘柄名"].astype(str) + "｜" + display_df["タイトル"].astype(str)).tolist()
    selected = st.selectbox("詳細を見る候補", choices, key=f"{key_prefix}_detail")
    row = display_df.iloc[choices.index(selected)]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("総合ランク", row.get("総合ランク", "-"))
    d2.metric("材料", f"{row.get('材料判定', '-')}/{row.get('重要度', '-')}")
    d3.metric("モメンタム", row.get("モメンタムスコア", "-"))
    d4.metric("テーマ", row.get("テーマ", "-"))

    st.write("### 翌日判断")
    st.write(row.get("翌日監視判断", ""))
    st.write("### 材料ポイント")
    st.write("好材料：", row.get("好材料ポイント", ""))
    st.write("悪材料：", row.get("悪材料ポイント", ""))
    st.write("本文数値：", row.get("本文から拾った数字", ""))
    st.write("### モメンタムポイント")
    st.write("強い理由：", row.get("強い理由", ""))
    st.write("見送り理由：", row.get("見送り理由", ""))
    if row.get("PDFリンク", ""):
        st.link_button("PDFを開く", row.get("PDFリンク"))


def run_material_momentum_screening(
    target_date,
    after_time,
    max_items,
    max_pdf_count,
    max_pages,
    title_keyword,
    only_good_material,
    momentum_preset_name,
    min_importance,
):
    source_df, error_msg = fetch_tdnet_disclosures_free(target_date, max_items=int(max_items))
    if source_df.empty:
        st.error("適時開示一覧を取得できませんでした。非公式API側の停止/遅延の可能性があります。")
        if error_msg:
            st.caption(error_msg)
        return pd.DataFrame()

    source_df = filter_disclosures_by_time(source_df, after_time)
    if title_keyword.strip():
        kw = title_keyword.strip()
        source_df = source_df[source_df["タイトル"].astype(str).str.contains(kw, case=False, na=False)]
    if "PDFリンク" in source_df.columns:
        source_df = source_df[source_df["PDFリンク"].astype(str).str.len() > 0]

    # 企業情報をJPX一覧から補完
    try:
        stock_master = load_jpx_list()
        stock_map = stock_master.set_index("コード").to_dict("index")
    except Exception:
        stock_map = {}

    params = MOMENTUM_PRESETS[momentum_preset_name]
    importance_rank = {"S": 4, "A": 3, "B": 2, "C": 1}
    min_imp_value = importance_rank.get(min_importance, 2)

    rows = []
    work_df = source_df.head(int(max_pdf_count)).reset_index(drop=True)
    total = len(work_df)
    progress = st.progress(0)
    status = st.empty()

    for i, row in work_df.iterrows():
        code = normalize_disclosure_code(row.get("コード", ""))
        company = row.get("銘柄名", "")
        title = row.get("タイトル", "")
        status.write(f"材料×モメンタム判定中：{code} {company} {i + 1}/{total}")

        pdf_text, pdf_error = extract_pdf_text_from_url(row.get("PDFリンク", ""), max_pages=int(max_pages))
        judgement = judge_disclosure_material(title=title, body_text=pdf_text, code=code, company=company)

        if only_good_material:
            if material_judgement_guard(judgement.get("材料判定", "")) not in ["好材料", "好悪混在", "要確認"]:
                progress.progress((i + 1) / total if total else 1.0)
                continue
            if importance_rank.get(judgement.get("重要度", "C"), 1) < min_imp_value:
                progress.progress((i + 1) / total if total else 1.0)
                continue

        meta = stock_map.get(code, {}) if code else {}
        name = meta.get("銘柄名", company)
        market = meta.get("市場", row.get("市場", ""))
        sector33 = meta.get("33業種", "")
        sector17 = meta.get("17業種", "")
        theme = meta.get("テーマ", judgement.get("テーマ", infer_theme(code, name, sector33, sector17)))

        momentum_result = None
        if code:
            try:
                price_df = download_price(code)
                momentum_result = analyze_momentum(
                    price_df=price_df,
                    code=code,
                    name=name,
                    market=market,
                    sector33=sector33,
                    sector17=sector17,
                    theme=theme,
                    params=params,
                    use_ma200_filter=False,
                )
            except Exception:
                momentum_result = None

        rank = calc_material_momentum_rank(judgement, momentum_result)
        out = row.to_dict()
        out.update({
            "コード": code,
            "銘柄名": name,
            "市場": market,
            "33業種": sector33,
            "17業種": sector17,
            "テーマ": theme,
        })
        out.update(judgement)
        out.update(rank)
        out["PDF読取エラー"] = pdf_error

        if momentum_result is None:
            out.update({
                "モメンタム判定": "判定不可",
                "モメンタム種類": "-",
                "モメンタムスコア": np.nan,
                "5日上昇率%": np.nan,
                "20日上昇率%": np.nan,
                "出来高倍率": np.nan,
                "20日高値距離%": np.nan,
                "損切り率%": np.nan,
                "強い理由": "株価データ未取得",
                "見送り理由": "株価データ未取得",
            })
        else:
            out.update({
                "モメンタム判定": momentum_result.get("買い候補", ""),
                "モメンタム種類": momentum_result.get("モメンタム種類", ""),
                "モメンタムスコア": momentum_result.get("スコア", np.nan),
                "5日上昇率%": momentum_result.get("5日上昇率%", np.nan),
                "20日上昇率%": momentum_result.get("20日上昇率%", np.nan),
                "出来高倍率": momentum_result.get("出来高倍率", np.nan),
                "20日高値距離%": momentum_result.get("20日高値距離%", np.nan),
                "損切り率%": momentum_result.get("損切り率%", np.nan),
                "強い理由": momentum_result.get("強い理由", ""),
                "見送り理由": momentum_result.get("見送り理由", ""),
                "上抜けライン": momentum_result.get("上抜けライン", np.nan),
                "損切りライン": momentum_result.get("損切りライン", np.nan),
            })
        rows.append(out)
        progress.progress((i + 1) / total if total else 1.0)

    status.empty()
    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    rank_order = {"S": 6, "A": 5, "A-要確認": 4, "B": 3, "B-要確認": 2, "C": 1, "見送り": 0}
    result_df["_ランク順"] = result_df["総合ランク"].map(rank_order).fillna(0)
    result_df = result_df.sort_values(["_ランク順", "総合スコア", "材料スコア", "モメンタムスコア"], ascending=[False, False, False, False])
    return result_df.reset_index(drop=True)


def render_material_momentum_tab(tab_key="material_momentum"):
    st.subheader("材料 × モメンタム")
    st.write("適時開示PDF本文の材料判定と、現在のモメンタム状態を合成して、翌日監視優先度を出します。")
    st.caption("無料版なのでAI読解ではなくPDF本文のルール判定です。重要候補は必ずPDF原文で確認してください。")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_date = st.date_input("開示日", value=datetime.now(JST).date(), key=f"{tab_key}_date")
    with c2:
        after_time_label = st.selectbox(
            "この時刻以降",
            ["00:00 全開示", "09:00 場中含む", "11:30 前引け後", "12:30 後場以降", "15:00 大引け後", "15:30 決算集中時間以降", "16:00 16時以降", "17:00 17時以降"],
            index=1,
            key=f"{tab_key}_after",
        )
        after_time = after_time_label.split()[0]
    with c3:
        max_items = st.number_input("取得する最大件数", min_value=20, max_value=1200, value=600, step=50, key=f"{tab_key}_max_items")
    with c4:
        max_pdf_count = st.number_input("PDF本文を読む件数", min_value=5, max_value=500, value=120, step=25, key=f"{tab_key}_max_pdf")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        max_pages = st.number_input("1PDFあたり読むページ数", min_value=1, max_value=30, value=8, step=1, key=f"{tab_key}_pages")
    with c6:
        momentum_preset = st.selectbox("モメンタム判定", ["普通", "厳しめ", "超厳しめ", "ゆるめ"], index=1, key=f"{tab_key}_momentum_preset")
    with c7:
        min_importance = st.selectbox("最低重要度", ["S", "A", "B", "C"], index=2, key=f"{tab_key}_min_imp")
    with c8:
        only_good_material = st.checkbox("好材料・要確認だけ処理", value=True, key=f"{tab_key}_only_good")

    title_keyword = st.text_input("タイトル絞り込み 任意", value="", placeholder="例：上方修正、増配、自己株式、受注", key=f"{tab_key}_kw")

    if st.button("材料×モメンタムを実行", type="primary", key=f"{tab_key}_run"):
        result_df = run_material_momentum_screening(
            target_date=target_date,
            after_time=after_time,
            max_items=int(max_items),
            max_pdf_count=int(max_pdf_count),
            max_pages=int(max_pages),
            title_keyword=title_keyword,
            only_good_material=only_good_material,
            momentum_preset_name=momentum_preset,
            min_importance=min_importance,
        )
        st.session_state[f"{tab_key}_results"] = result_df

    if f"{tab_key}_results" in st.session_state:
        render_material_momentum_result(st.session_state[f"{tab_key}_results"], key_prefix=tab_key)



# =========================
# 材料インパクトランキング
# =========================

IMPACT_THEME_BONUS_KEYWORDS = [
    "AI", "データセンター", "半導体", "防衛", "宇宙", "衛星", "ロケット",
    "電力", "原発", "送電", "HDD", "ストレージ", "ビットコイン", "暗号資産",
    "レアアース", "蓄電", "電池", "バイオ", "治験", "FDA",
]

IMPACT_DIRECT_BUY_PATTERNS = [
    r"TOB|公開買付|MBO|応募推奨|賛同の意見表明",
    r"上方修正.*増配|増配.*上方修正",
    r"黒字転換",
    r"自己株式.*取得.*発行済株式.*[3-9]\.\d+%|自己株式.*取得.*発行済株式.*[3-9]％",
]

IMPACT_WAIT_PATTERNS = [
    r"業務提携|資本業務提携|共同開発|新製品|販売開始|採用|導入|中期経営計画|成長戦略|月次",
]


def _impact_contains(text, patterns):
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            return True
    return False


def calc_material_impact_rank(judgement, title="", body_text=""):
    """
    モメンタムを使わず、PDF本文から材料そのものの株価インパクトをランキング化する。
    引け後・場中どちらの開示でも、まず「翌日監視すべき材料か」を見る用途。
    """
    material = judgement.get("材料判定", "")
    guard = material_judgement_guard(material)
    importance = str(judgement.get("重要度", "C"))
    material_score = float(judgement.get("材料スコア", 0) or 0)
    good_points = str(judgement.get("好材料ポイント", ""))
    bad_points = str(judgement.get("悪材料ポイント", ""))
    numbers = str(judgement.get("本文から拾った数字", ""))
    theme = str(judgement.get("テーマ", ""))
    joined = re.sub(r"\s+", " ", f"{title}\n{body_text[:2500]}\n{good_points}\n{bad_points}\n{numbers}\n{theme}")

    score = material_score + material_importance_points(importance)
    reasons = []

    if importance in ["S", "A"]:
        reasons.append(f"材料重要度が{importance}")
    if good_points:
        reasons.append(f"好材料ポイント：{good_points}")
    if numbers:
        reasons.append("本文内に業績・配当・金額などの数値あり")

    # 直接株価に効きやすい材料を加点
    if re.search(r"TOB|公開買付|MBO|応募推奨|賛同の意見表明", joined, flags=re.IGNORECASE):
        score += 80
        reasons.append("TOB/MBO系で需給インパクトが大きい")
    if re.search(r"上方修正|上方に修正|業績予想.*増額|予想.*上回", joined, flags=re.IGNORECASE):
        score += 30
        reasons.append("業績上方修正")
    if re.search(r"営業利益|経常利益|純利益", joined) and re.search(r"[+＋]\s*([3-9][0-9]|[1-9][0-9]{2,})\s*[%％]|([3-9][0-9]|[1-9][0-9]{2,})\s*[%％].*(増|上)", joined):
        score += 25
        reasons.append("利益インパクトが大きい可能性")
    if re.search(r"増配|復配|配当.*引き上げ|配当予想.*増額", joined, flags=re.IGNORECASE):
        score += 22
        reasons.append("増配/復配")
    if re.search(r"自己株式の取得(?!状況)|自己株式取得に係る事項の決定|自己株式.*消却", joined, flags=re.IGNORECASE):
        score += 20
        reasons.append("自社株買い/消却")
    if re.search(r"黒字転換|営業黒字|経常黒字", joined):
        score += 28
        reasons.append("黒字転換")
    if re.search(r"大型受注|受注|契約締結|基本契約|販売契約|採用|導入決定", joined):
        score += 20
        reasons.append("受注・契約・採用系")
    if re.search(r"資本業務提携|業務提携|戦略的業務提携|共同開発|事業協力", joined):
        score += 18
        reasons.append("提携/共同開発")
    if re.search(r"株式分割", joined):
        score += 15
        reasons.append("株式分割")
    if any(k.lower() in joined.lower() for k in IMPACT_THEME_BONUS_KEYWORDS):
        score += 12
        reasons.append("テーマ性あり")

    # 悪材料・希薄化は強めに減点
    if re.search(r"MSワラント|行使価額修正|新株予約権|第三者割当|公募増資|希薄化|新株式発行", joined, flags=re.IGNORECASE):
        score -= 45
        reasons.append("希薄化リスクあり")
    if re.search(r"下方修正|減配|無配|赤字転落|減損損失|特別損失|不適切会計|決算発表.*延期", joined, flags=re.IGNORECASE):
        score -= 35
        reasons.append("悪材料要素あり")

    # ランク決定
    if guard == "悪材料":
        if abs(material_score) >= 75 or importance == "S":
            impact_rank = "見送りS"
            next_action = "買い対象外。悪材料インパクトが大きい可能性が高い"
        elif abs(material_score) >= 40 or importance == "A":
            impact_rank = "見送りA"
            next_action = "基本見送り。保有時はリスク確認優先"
        else:
            impact_rank = "見送り"
            next_action = "買い対象外。内容確認のみ"
        trade_type = "見送り"
    elif guard == "好悪混在":
        if score >= 85:
            impact_rank = "A-要確認"
            next_action = "好材料も強いが悪材料部分を原文確認。飛びつきは慎重"
        else:
            impact_rank = "B-要確認"
            next_action = "好悪混在。翌日監視は可だが、希薄化・減配・損失を確認"
        trade_type = "内容確認後に判断"
    elif guard == "好材料":
        if score >= 115 or _impact_contains(joined, IMPACT_DIRECT_BUY_PATTERNS):
            impact_rank = "S"
            next_action = "寄り前から最優先監視。気配が強すぎる場合は無理せず押し目も検討"
            trade_type = "寄り前最優先監視"
        elif score >= 78:
            impact_rank = "A"
            next_action = "翌日監視候補。寄り付き位置・出来高・板を確認"
            trade_type = "寄り後確認"
        elif score >= 42:
            impact_rank = "B"
            next_action = "材料は確認価値あり。急騰追いより押し目待ち向き"
            trade_type = "押し目待ち"
        else:
            impact_rank = "C"
            next_action = "単独材料としては弱め。監視リスト程度"
            trade_type = "軽く確認"
    elif guard == "要確認":
        if score >= 70:
            impact_rank = "B-要確認"
            next_action = "タイトルだけでは不明だが本文に数値あり。原文確認"
        else:
            impact_rank = "C"
            next_action = "要確認。売買材料にするには弱い"
        trade_type = "内容確認"
    else:
        impact_rank = "C"
        next_action = "中立寄り。基本は売買材料にしない"
        trade_type = "見送り寄り"

    material_type = good_points or bad_points or "要確認"
    impact_reason = " / ".join(list(dict.fromkeys([r for r in reasons if r]))[:6])
    if not impact_reason:
        impact_reason = "目立つ株価インパクト材料は限定的"

    return {
        "材料インパクトランク": impact_rank,
        "インパクトスコア": round(float(score), 1),
        "材料タイプ": material_type,
        "株価インパクト理由": impact_reason,
        "翌日対応": next_action,
        "売買方針": trade_type,
    }


def render_material_impact_result(result_df, key_prefix="material_impact"):
    if result_df is None or result_df.empty:
        st.warning("条件に合う材料インパクト候補はありませんでした。時刻・件数・表示条件を変えてください。")
        return

    st.subheader("材料インパクトランキング")
    st.write(f"表示件数：{len(result_df)}")

    c1, c2, c3 = st.columns(3)
    with c1:
        rank_filter = st.multiselect(
            "表示ランク",
            ["S", "A", "A-要確認", "B", "B-要確認", "C", "見送りS", "見送りA", "見送り"],
            default=["S", "A", "A-要確認", "B", "B-要確認"],
            key=f"{key_prefix}_rank_filter",
        )
    with c2:
        material_filter = st.multiselect(
            "材料判定",
            ["好材料", "好悪混在", "要確認", "中立", "悪材料"],
            default=["好材料", "好悪混在", "要確認"],
            key=f"{key_prefix}_material_filter",
        )
    with c3:
        theme_keyword = st.text_input("テーマ/タイトル絞り込み", value="", placeholder="例：半導体、AI、防衛、宇宙、上方修正", key=f"{key_prefix}_theme_kw")

    display_df = result_df.copy()
    if rank_filter:
        display_df = display_df[display_df["材料インパクトランク"].isin(rank_filter)]
    if material_filter:
        display_df = display_df[display_df["材料判定"].isin(material_filter)]
    if theme_keyword.strip():
        kw = theme_keyword.strip()
        display_df = display_df[
            display_df["テーマ"].astype(str).str.contains(kw, case=False, na=False)
            | display_df["銘柄名"].astype(str).str.contains(kw, case=False, na=False)
            | display_df["タイトル"].astype(str).str.contains(kw, case=False, na=False)
            | display_df["材料タイプ"].astype(str).str.contains(kw, case=False, na=False)
        ]

    order = [
        "順位", "材料インパクトランク", "インパクトスコア", "時刻", "コード", "銘柄名", "テーマ", "タイトル",
        "材料判定", "重要度", "材料タイプ", "株価インパクト理由", "本文から拾った数字",
        "翌日対応", "売買方針", "好材料ポイント", "悪材料ポイント", "PDFリンク", "PDF読取エラー",
    ]
    cols = [c for c in order if c in display_df.columns]
    st.dataframe(
        display_df[cols],
        use_container_width=True,
        hide_index=True,
        column_config={"PDFリンク": st.column_config.LinkColumn("PDFリンク")} if "PDFリンク" in cols else None,
    )

    csv = display_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "CSVダウンロード",
        data=csv,
        file_name="material_impact_ranking.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )

    if display_df.empty:
        return

    st.subheader("詳細確認")
    choices = (display_df["材料インパクトランク"].astype(str) + "｜" + display_df["コード"].astype(str) + " " + display_df["銘柄名"].astype(str) + "｜" + display_df["タイトル"].astype(str)).tolist()
    selected = st.selectbox("詳細を見る材料", choices, key=f"{key_prefix}_detail")
    row = display_df.iloc[choices.index(selected)]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("インパクトランク", row.get("材料インパクトランク", "-"))
    d2.metric("材料判定", f"{row.get('材料判定', '-')}/{row.get('重要度', '-')}")
    d3.metric("スコア", row.get("インパクトスコア", "-"))
    d4.metric("テーマ", row.get("テーマ", "-"))

    st.write("### 翌日対応")
    st.write(row.get("翌日対応", ""))
    st.write("### 株価インパクト理由")
    st.write(row.get("株価インパクト理由", ""))
    st.write("### 本文から拾った数字")
    st.write(row.get("本文から拾った数字", ""))
    st.write("### 好材料/悪材料ポイント")
    st.write("好材料：", row.get("好材料ポイント", ""))
    st.write("悪材料：", row.get("悪材料ポイント", ""))
    if row.get("PDFリンク", ""):
        st.link_button("PDFを開く", row.get("PDFリンク"))


def run_material_impact_screening(
    target_date,
    after_time,
    max_items,
    max_pdf_count,
    max_pages,
    title_keyword,
    show_bad_material,
    min_rank,
):
    source_df, error_msg = fetch_tdnet_disclosures_free(target_date, max_items=int(max_items))
    if source_df.empty:
        st.error("適時開示一覧を取得できませんでした。非公式API側の停止/遅延の可能性があります。")
        if error_msg:
            st.caption(error_msg)
        return pd.DataFrame()

    source_df = filter_disclosures_by_time(source_df, after_time)
    if title_keyword.strip():
        kw = title_keyword.strip()
        source_df = source_df[source_df["タイトル"].astype(str).str.contains(kw, case=False, na=False)]
    if "PDFリンク" in source_df.columns:
        source_df = source_df[source_df["PDFリンク"].astype(str).str.len() > 0]

    try:
        stock_master = load_jpx_list()
        stock_map = stock_master.set_index("コード").to_dict("index")
    except Exception:
        stock_map = {}

    rows = []
    work_df = source_df.head(int(max_pdf_count)).reset_index(drop=True)
    total = len(work_df)
    progress = st.progress(0)
    status = st.empty()

    for i, row in work_df.iterrows():
        code = normalize_disclosure_code(row.get("コード", ""))
        company = row.get("銘柄名", "")
        title = row.get("タイトル", "")
        status.write(f"材料インパクト判定中：{code} {company} {i + 1}/{total}")

        pdf_text, pdf_error = extract_pdf_text_from_url(row.get("PDFリンク", ""), max_pages=int(max_pages))
        judgement = judge_disclosure_material(title=title, body_text=pdf_text, code=code, company=company)
        impact = calc_material_impact_rank(judgement, title=title, body_text=pdf_text)

        guard = material_judgement_guard(judgement.get("材料判定", ""))
        if (not show_bad_material) and guard == "悪材料":
            progress.progress((i + 1) / total if total else 1.0)
            continue

        meta = stock_map.get(code, {}) if code else {}
        name = meta.get("銘柄名", company)
        market = meta.get("市場", row.get("市場", ""))
        sector33 = meta.get("33業種", "")
        sector17 = meta.get("17業種", "")
        theme = meta.get("テーマ", judgement.get("テーマ", infer_theme(code, name, sector33, sector17)))
        if is_bad_theme_text(theme):
            theme = judgement.get("テーマ", "個別テーマ要確認")

        out = row.to_dict()
        out.update({
            "コード": code,
            "銘柄名": name,
            "市場": market,
            "33業種": sector33,
            "17業種": sector17,
            "テーマ": theme,
        })
        out.update(judgement)
        out.update(impact)
        out["PDF読取エラー"] = pdf_error
        rows.append(out)
        progress.progress((i + 1) / total if total else 1.0)

    status.empty()
    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    rank_order = {"S": 9, "A": 8, "A-要確認": 7, "B": 6, "B-要確認": 5, "C": 4, "見送りS": 3, "見送りA": 2, "見送り": 1}
    result_df["_ランク順"] = result_df["材料インパクトランク"].map(rank_order).fillna(0)
    result_df = result_df.sort_values(["_ランク順", "インパクトスコア", "材料スコア", "開示日時"], ascending=[False, False, False, False])
    result_df = result_df.reset_index(drop=True)
    result_df["順位"] = np.arange(1, len(result_df) + 1)

    if min_rank != "全部表示":
        min_order = rank_order.get(min_rank, 0)
        result_df = result_df[result_df["_ランク順"] >= min_order].reset_index(drop=True)
        result_df["順位"] = np.arange(1, len(result_df) + 1)

    return result_df


def render_material_impact_tab(tab_key="material_impact"):
    st.subheader("材料インパクトランキング")
    st.write("適時開示PDF本文を読み、モメンタム条件は使わずに『材料そのものが株価に効きそうか』をランキングします。")
    st.caption("引け後材料・場中材料の翌日監視リスト作成向けです。無料版なのでAIではなくPDF本文のルール判定です。")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_date = st.date_input("開示日", value=datetime.now(JST).date(), key=f"{tab_key}_date")
    with c2:
        after_time_label = st.selectbox(
            "この時刻以降",
            ["00:00 全開示", "09:00 場中含む", "11:30 前引け後", "12:30 後場以降", "15:00 大引け後", "15:30 決算集中時間以降", "16:00 16時以降", "17:00 17時以降"],
            index=1,
            key=f"{tab_key}_after",
        )
        after_time = after_time_label.split()[0]
    with c3:
        max_items = st.number_input("取得する最大件数", min_value=20, max_value=1200, value=600, step=50, key=f"{tab_key}_max_items")
    with c4:
        max_pdf_count = st.number_input("PDF本文を読む件数", min_value=5, max_value=500, value=150, step=25, key=f"{tab_key}_max_pdf")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        max_pages = st.number_input("1PDFあたり読むページ数", min_value=1, max_value=30, value=8, step=1, key=f"{tab_key}_pages")
    with c6:
        min_rank = st.selectbox("最低表示ランク", ["S", "A", "A-要確認", "B", "B-要確認", "C", "全部表示"], index=3, key=f"{tab_key}_min_rank")
    with c7:
        show_bad_material = st.checkbox("悪材料も表示", value=False, key=f"{tab_key}_show_bad")
    with c8:
        st.caption("まずはB以上表示がおすすめ。件数が多ければA以上にしてください。")

    title_keyword = st.text_input("タイトル絞り込み 任意", value="", placeholder="例：上方修正、増配、自己株式、受注、提携", key=f"{tab_key}_kw")

    if st.button("材料インパクトランキングを実行", type="primary", key=f"{tab_key}_run"):
        result_df = run_material_impact_screening(
            target_date=target_date,
            after_time=after_time,
            max_items=int(max_items),
            max_pdf_count=int(max_pdf_count),
            max_pages=int(max_pages),
            title_keyword=title_keyword,
            show_bad_material=show_bad_material,
            min_rank=min_rank,
        )
        st.session_state[f"{tab_key}_results"] = result_df

    if f"{tab_key}_results" in st.session_state:
        render_material_impact_result(st.session_state[f"{tab_key}_results"], key_prefix=tab_key)

# =========================
# テーマ別資金流入ランキング
# =========================

def main_theme_label(theme_text):
    text = str(theme_text).strip()
    if is_bad_theme_text(text):
        return "個別テーマ要確認"
    first = re.split(r"/|／|,|、", text)[0].strip()
    if is_bad_theme_text(first):
        return "個別テーマ要確認"
    return first or "個別テーマ要確認"


def _safe_mean(series, default=0.0):
    try:
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return default
        return float(s.mean())
    except Exception:
        return default


def _safe_count_condition(df, col, condition_func):
    try:
        if col not in df.columns:
            return 0
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return int(condition_func(s).sum())
    except Exception:
        return 0


def calc_theme_flow_score(g):
    candidate_count = int((g["買い候補"] == "候補").sum()) if "買い候補" in g else 0
    watch_count = int((g["買い候補"] == "監視").sum()) if "買い候補" in g else 0
    avg_score = _safe_mean(g["スコア"]) if "スコア" in g else 0
    avg_ret5 = _safe_mean(g["5日上昇率%"]) if "5日上昇率%" in g else 0
    avg_ret20 = _safe_mean(g["20日上昇率%"]) if "20日上昇率%" in g else 0
    avg_vol = _safe_mean(g["出来高倍率"]) if "出来高倍率" in g else 0
    highzone = _safe_count_condition(g, "20日高値距離%", lambda s: s <= 3)

    # 個別銘柄名ではなく、テーマ全体の勢いを点数化する。
    # 候補/監視の数、上昇率、出来高、高値圏の銘柄数を重視。
    score = (
        candidate_count * 22
        + watch_count * 10
        + avg_score * 0.75
        + max(avg_ret5, 0) * 1.8
        + max(avg_ret20, 0) * 0.8
        + min(max(avg_vol, 0), 6) * 10
        + highzone * 6
    )
    return round(score, 1)


def theme_flow_grade(flow_score):
    if flow_score >= 120:
        return "S"
    if flow_score >= 90:
        return "A"
    if flow_score >= 65:
        return "B"
    if flow_score >= 40:
        return "C"
    return "D"


def theme_flow_type(candidate_count, watch_count, avg_ret5, avg_ret20, avg_vol, highzone_count, total_count):
    if candidate_count >= 2 and avg_vol >= 2.0 and highzone_count >= 2:
        return "資金集中・高値追い"
    if avg_ret5 >= 10 and avg_vol >= 2.0:
        return "短期急騰・出来高集中"
    if avg_ret20 >= 18 and highzone_count >= max(1, total_count // 3):
        return "中期上昇・高値圏"
    if candidate_count >= 1 and watch_count >= 1:
        return "候補増加・監視強め"
    if watch_count >= 2:
        return "監視銘柄増加"
    return "弱め・要確認"


# =========================
# テーマ流入理由：ネットニュース検索
# =========================

NEWS_IMPACT_KEYWORDS = [
    "生成AI", "AI", "データセンター", "GPU", "HBM", "半導体", "設備投資", "増産",
    "電力需要", "送電網", "変圧器", "液冷", "冷却", "防衛費", "防衛装備", "ドローン",
    "ミサイル", "衛星", "ロケット", "宇宙", "国策", "補助金", "政策", "受注", "大型受注",
    "提携", "資本業務提携", "TOB", "値上げ", "価格上昇", "需給逼迫", "供給不足",
    "レアアース", "金", "銅", "原子力", "原発", "再稼働", "造船", "海運", "運賃",
    "バイオ", "治験", "承認", "インバウンド", "訪日", "円安", "ビットコイン", "暗号資産",
]


def build_theme_news_queries(theme, news_days=7):
    """
    テーマ名からGoogle News RSS向けの検索語を作る。
    完全なニュース検索APIではなく、無料で使えるRSS検索を利用する。
    """
    t = str(theme)
    d = int(news_days)

    if "半導体" in t:
        return [
            f'半導体 AI HBM 設備投資 日本株 when:{d}d',
            f'半導体製造装置 生成AI 需要 when:{d}d',
        ]
    if "AI" in t or "データ" in t or t == "AI":
        return [
            f'生成AI データセンター 投資 電力需要 when:{d}d',
            f'AI データセンター 日本企業 半導体 when:{d}d',
        ]
    if "電線" in t or "電力インフラ" in t or "電力" in t:
        return [
            f'電線 データセンター 電力需要 送電網 when:{d}d',
            f'電力インフラ 変圧器 送電網 投資 when:{d}d',
        ]
    if "防衛" in t:
        return [
            f'防衛費 防衛装備 ドローン 日本企業 when:{d}d',
            f'防衛 受注 ミサイル レーダー when:{d}d',
        ]
    if "宇宙" in t:
        return [
            f'宇宙 衛星 ロケット 防衛省 契約 when:{d}d',
            f'宇宙ビジネス 衛星 日本企業 when:{d}d',
        ]
    if "MLCC" in t or "電子部品" in t:
        return [
            f'MLCC 電子部品 AIサーバー 需要 when:{d}d',
            f'コンデンサ 電子部品 価格 需給 when:{d}d',
        ]
    if "HDD" in t or "ストレージ" in t:
        return [
            f'HDD ストレージ AI データセンター 需給 when:{d}d',
            f'ハードディスク ニアライン 需要 when:{d}d',
        ]
    if "原発" in t or "原子力" in t:
        return [
            f'原発 再稼働 電力政策 日本 when:{d}d',
            f'原子力 電力需要 データセンター when:{d}d',
        ]
    if "造船" in t or "海運" in t:
        return [
            f'造船 受注 海運 市況 when:{d}d',
            f'船舶 運賃 造船 日本企業 when:{d}d',
        ]
    if "バイオ" in t or "医薬" in t:
        return [
            f'バイオ 医薬品 治験 承認 日本企業 when:{d}d',
            f'創薬 提携 ライセンス契約 when:{d}d',
        ]
    if "インバウンド" in t or "小売" in t:
        return [
            f'インバウンド 訪日消費 小売 百貨店 when:{d}d',
            f'訪日客 円安 消費 小売 when:{d}d',
        ]
    if "金融" in t or "暗号" in t or "ビットコイン" in t:
        return [
            f'ビットコイン 暗号資産 ETF 日本株 when:{d}d',
            f'銀行 金利 証券 株式市場 when:{d}d',
        ]
    if "資源" in t or "レアアース" in t:
        return [
            f'レアアース 資源 価格 上昇 日本企業 when:{d}d',
            f'銅 金 非鉄 金属 価格 when:{d}d',
        ]

    return [f'{t} 日本株 関連 ニュース when:{d}d']


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_google_news_rss(query, max_items=5):
    """
    Google News RSS検索で関連ニュースを取得する。
    無料・APIキーなしで使うため、結果はGoogle News RSSの取得状況に依存する。
    """
    if BeautifulSoup is None:
        return []

    url = "https://news.google.com/rss/search?q=" + quote(query) + "&hl=ja&gl=JP&ceid=JP:ja"
    try:
        res = requests.get(url, headers=REQUEST_HEADERS, timeout=12)
        if res.status_code != 200:
            return []
        soup = BeautifulSoup(res.content, "xml")
        items = soup.find_all("item")[: int(max_items)]
        articles = []
        for item in items:
            title = item.title.get_text(" ", strip=True) if item.title else ""
            link = item.link.get_text(" ", strip=True) if item.link else ""
            source = item.source.get_text(" ", strip=True) if item.source else ""
            pub_raw = item.pubDate.get_text(" ", strip=True) if item.pubDate else ""
            pub_dt = None
            if pub_raw:
                try:
                    pub_dt = parsedate_to_datetime(pub_raw)
                    if pub_dt and pub_dt.tzinfo is not None:
                        pub_dt = pub_dt.astimezone(timezone(timedelta(hours=9)))
                except Exception:
                    pub_dt = None
            articles.append({
                "title": title,
                "source": source,
                "published": pub_dt.strftime("%m/%d %H:%M") if pub_dt else "",
                "link": link,
            })
        return articles
    except Exception:
        return []


def dedupe_news_articles(articles):
    seen = set()
    out = []
    for a in articles:
        title = str(a.get("title", "")).strip()
        if not title:
            continue
        key = re.sub(r"\s+", "", title.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def extract_news_keywords(articles):
    text = " ".join(str(a.get("title", "")) for a in articles)
    hits = []
    for kw in NEWS_IMPACT_KEYWORDS:
        if kw.lower() in text.lower():
            hits.append(kw)
    return hits[:8]


def build_web_theme_reason(theme, articles):
    """
    ニュースタイトルから、テーマに資金が入りそうな背景を短く作る。
    個別銘柄数ではなく、外部ニュース由来の理由を表示する。
    """
    if not articles:
        return "ネットニュースから明確な関連材料を取得できず。テーマ自体の値動きは強い可能性があるが、理由は要確認。", "", 0

    keywords = extract_news_keywords(articles)
    titles = [str(a.get("title", "")).strip() for a in articles if str(a.get("title", "")).strip()]

    if keywords:
        material = "、".join(keywords)
        reason = f"直近ニュースで「{material}」が目立つ。{theme}への資金流入は、これらの材料・連想買いが背景になっている可能性。"
    else:
        # キーワードに引っかからない場合でも、ニュース見出しの共通話題を根拠にする
        sample = " / ".join(titles[:2])
        reason = f"直近ニュースで関連見出しが複数確認できる。主な話題：{sample}"

    # 表に載せる根拠は長くなりすぎないよう2本まで
    evidence_parts = []
    for a in articles[:2]:
        title = a.get("title", "")
        source = a.get("source", "")
        published = a.get("published", "")
        if title:
            suffix = f"（{source} {published}）" if source or published else ""
            evidence_parts.append(f"{title}{suffix}")
    evidence = " / ".join(evidence_parts)
    return reason, evidence, len(articles)


def enrich_theme_ranking_with_web_reasons(ranking_df, news_days=7, news_limit=5, max_themes=15):
    if ranking_df is None or ranking_df.empty:
        return ranking_df

    out = ranking_df.copy()
    out["ネット流入理由"] = ""
    out["ニュース材料"] = ""
    out["関連ニュース件数"] = 0

    max_themes = min(int(max_themes), len(out))
    for idx in out.index[:max_themes]:
        theme = str(out.at[idx, "テーマ"])
        queries = build_theme_news_queries(theme, news_days=news_days)
        articles = []
        for q in queries:
            articles.extend(fetch_google_news_rss(q, max_items=news_limit))
        articles = dedupe_news_articles(articles)[: int(news_limit)]
        reason, evidence, count = build_web_theme_reason(theme, articles)
        out.at[idx, "ネット流入理由"] = reason
        out.at[idx, "ニュース材料"] = evidence
        out.at[idx, "関連ニュース件数"] = count

    # 検索しなかった下位テーマは空欄ではなく明記
    if max_themes < len(out):
        out.loc[out.index[max_themes:], "ネット流入理由"] = "上位テーマのみニュース検索対象。必要なら検索する上位テーマ数を増やしてください。"

    # 既存の「流入理由」はニュース由来を優先。ニュースが取れない場合は補助コメントを入れる。
    out["流入理由"] = out["ネット流入理由"].where(out["ネット流入理由"].astype(str).str.len() > 0, out.get("流入理由", ""))
    return out


def build_theme_flow_reason(row):
    reasons = []

    candidate_count = int(row.get("候補数", 0))
    watch_count = int(row.get("監視数", 0))
    avg_score = float(row.get("平均スコア", 0) or 0)
    avg_ret5 = float(row.get("平均5日上昇率%", 0) or 0)
    avg_ret20 = float(row.get("平均20日上昇率%", 0) or 0)
    avg_vol = float(row.get("平均出来高倍率", 0) or 0)
    highzone = int(row.get("20日高値3%以内数", 0) or 0)

    if candidate_count > 0:
        reasons.append(f"買い候補が{candidate_count}件")
    if watch_count > 0:
        reasons.append(f"監視候補が{watch_count}件")
    if avg_score >= 75:
        reasons.append(f"平均スコアが高い({avg_score:.1f})")
    elif avg_score >= 60:
        reasons.append(f"平均スコアが良好({avg_score:.1f})")
    if avg_ret5 >= 10:
        reasons.append(f"直近5日で強い上昇(+{avg_ret5:.1f}%)")
    elif avg_ret5 >= 5:
        reasons.append(f"直近5日が堅調(+{avg_ret5:.1f}%)")
    if avg_ret20 >= 20:
        reasons.append(f"20日上昇率が大きい(+{avg_ret20:.1f}%)")
    elif avg_ret20 >= 10:
        reasons.append(f"20日上昇率が良好(+{avg_ret20:.1f}%)")
    if avg_vol >= 2.5:
        reasons.append(f"出来高が大きく増加({avg_vol:.1f}倍)")
    elif avg_vol >= 1.5:
        reasons.append(f"出来高増加({avg_vol:.1f}倍)")
    if highzone > 0:
        reasons.append(f"20日高値3%以内が{highzone}件")

    if not reasons:
        return "明確な資金流入シグナルは弱い。条件を緩めて監視、または別テーマ優先。"

    return " / ".join(reasons)


def build_theme_flow_ranking(result_df, min_stocks=2):
    if result_df is None or result_df.empty:
        return pd.DataFrame()

    df = result_df.copy()
    if "テーマ" not in df.columns:
        return pd.DataFrame()

    df["主テーマ"] = df["テーマ"].apply(main_theme_label)
    rows = []

    for theme, g in df.groupby("主テーマ"):
        # 数字コードや銘柄コード、不明テーマはランキングから除外する。
        # 「何のテーマか分からない行」を出さないため。
        if theme == "個別テーマ要確認" or is_bad_theme_text(theme):
            continue
        if len(g) < min_stocks:
            continue

        candidate_count = int((g["買い候補"] == "候補").sum()) if "買い候補" in g else 0
        watch_count = int((g["買い候補"] == "監視").sum()) if "買い候補" in g else 0
        avg_score = round(_safe_mean(g["スコア"]), 1) if "スコア" in g else 0
        avg_ret5 = round(_safe_mean(g["5日上昇率%"]), 2) if "5日上昇率%" in g else 0
        avg_ret20 = round(_safe_mean(g["20日上昇率%"]), 2) if "20日上昇率%" in g else 0
        avg_vol = round(_safe_mean(g["出来高倍率"]), 2) if "出来高倍率" in g else 0
        highzone_count = _safe_count_condition(g, "20日高値距離%", lambda s: s <= 3)
        near_high_count = _safe_count_condition(g, "20日高値距離%", lambda s: s <= 5)

        row = {
            "テーマ": theme,
            "対象銘柄数": len(g),
            "候補数": candidate_count,
            "監視数": watch_count,
            "候補監視合計": candidate_count + watch_count,
            "平均スコア": avg_score,
            "平均5日上昇率%": avg_ret5,
            "平均20日上昇率%": avg_ret20,
            "平均出来高倍率": avg_vol,
            "20日高値3%以内数": highzone_count,
            "20日高値5%以内数": near_high_count,
        }
        row["資金流入スコア"] = calc_theme_flow_score(g)
        row["流入ランク"] = theme_flow_grade(row["資金流入スコア"])
        row["流入タイプ"] = theme_flow_type(
            candidate_count=candidate_count,
            watch_count=watch_count,
            avg_ret5=avg_ret5,
            avg_ret20=avg_ret20,
            avg_vol=avg_vol,
            highzone_count=highzone_count,
            total_count=len(g),
        )
        row["流入理由"] = build_theme_flow_reason(row)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["資金流入スコア", "候補監視合計", "平均スコア", "平均出来高倍率"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    out.insert(0, "順位", range(1, len(out) + 1))
    return out


def run_theme_flow_screening(market_choice, preset_name, max_scan, wait_sec, use_ma200_filter):
    params = MOMENTUM_PRESETS[preset_name]
    try:
        all_stocks = load_jpx_list()
    except Exception as e:
        st.error(f"JPXの銘柄リスト取得に失敗しました: {e}")
        return pd.DataFrame(), 0

    target_stocks = filter_market(all_stocks, market_choice)
    if max_scan > 0:
        target_stocks = target_stocks.head(max_scan)

    total = len(target_stocks)
    results = []
    failed_count = 0
    progress_bar = st.progress(0)
    status_area = st.empty()

    for n, (_, row) in enumerate(target_stocks.iterrows(), start=1):
        code = row["コード"]
        name = row["銘柄名"]
        market = row["市場"]
        sector33 = row.get("33業種", "")
        sector17 = row.get("17業種", "")
        theme = row.get("テーマ", infer_theme(code, name, sector33, sector17))
        status_area.write(f"テーマ資金流入確認中：{n}/{total} {code} {name}")
        try:
            price_df = download_price(code)
            result = analyze_momentum(
                price_df=price_df,
                code=code,
                name=name,
                market=market,
                sector33=sector33,
                sector17=sector17,
                theme=theme,
                params=params,
                use_ma200_filter=use_ma200_filter,
            )
            if result is not None:
                results.append(result)
        except Exception:
            failed_count += 1
        progress_bar.progress(n / total if total else 1.0)
        if wait_sec > 0:
            time.sleep(wait_sec)

    status_area.write("テーマ資金流入ランキング作成完了")
    if not results:
        return pd.DataFrame(), failed_count
    result_df = pd.DataFrame(results)
    return result_df.reset_index(drop=True), failed_count


def render_theme_flow_tab(tab_key="theme_flow"):
    st.subheader("テーマ資金流入ランキング")
    st.write("個別銘柄名は表示せず、どのテーマに資金が入っているかをランキング形式で表示します。テーマは銘柄コードではなく、JPXの業種区分・銘柄名・テーマ辞書から推定します。流入理由はネットニュースの見出しから探します。")

    c1, c2, c3 = st.columns(3)
    with c1:
        preset_name = st.selectbox("判定の厳しさ", ["普通", "厳しめ", "超厳しめ", "ゆるめ"], index=0, key=f"{tab_key}_preset")
    with c2:
        min_stocks = st.number_input("テーマ集計の最低銘柄数", min_value=1, max_value=30, value=2, step=1, key=f"{tab_key}_min_stocks")
    with c3:
        only_candidate_watch = st.checkbox("候補・監視があるテーマのみ表示", value=True, key=f"{tab_key}_only_cw")

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        use_web_reason = st.checkbox("ネットニュースで流入理由を探す", value=True, key=f"{tab_key}_web_reason")
    with c5:
        news_days = st.selectbox("ニュース検索期間", [1, 3, 7, 14], index=2, key=f"{tab_key}_news_days")
    with c6:
        max_news_themes = st.number_input("ニュース検索する上位テーマ数", min_value=3, max_value=50, value=15, step=1, key=f"{tab_key}_max_news_themes")
    with c7:
        news_limit = st.number_input("1テーマあたりニュース件数", min_value=2, max_value=10, value=5, step=1, key=f"{tab_key}_news_limit")

    st.caption("流入理由は、候補件数ではなくGoogle News RSS等のネットニュース見出しから自動作成します。無料取得のため、ニュース側の配信状況により取得できない場合があります。")

    if st.button("テーマ資金流入ランキングを実行", type="primary", key=f"{tab_key}_run"):
        result_df, failed_count = run_theme_flow_screening(
            market_choice=market_choice,
            preset_name=preset_name,
            max_scan=max_scan,
            wait_sec=wait_sec,
            use_ma200_filter=use_ma200_filter,
        )
        st.session_state[f"{tab_key}_stock_results"] = result_df
        st.session_state[f"{tab_key}_failed_count"] = failed_count
        ranking_df = build_theme_flow_ranking(result_df, min_stocks=int(min_stocks))
        if use_web_reason and ranking_df is not None and not ranking_df.empty:
            with st.spinner("ネットニュースからテーマ流入理由を検索中..."):
                ranking_df = enrich_theme_ranking_with_web_reasons(
                    ranking_df,
                    news_days=int(news_days),
                    news_limit=int(news_limit),
                    max_themes=int(max_news_themes),
                )
        st.session_state[f"{tab_key}_ranking"] = ranking_df

    if f"{tab_key}_ranking" not in st.session_state:
        return

    ranking_df = st.session_state[f"{tab_key}_ranking"].copy()
    failed_count = st.session_state.get(f"{tab_key}_failed_count", 0)

    st.subheader("テーマランキング")
    st.write(f"取得失敗数：{failed_count}")
    if ranking_df.empty:
        st.warning("ランキングを作れるテーマがありませんでした。確認銘柄数を増やすか、最低銘柄数を下げてください。")
        return

    display_rank = ranking_df.copy()
    if only_candidate_watch and "候補監視合計" in display_rank.columns:
        filtered = display_rank[display_rank["候補監視合計"] > 0]
        if not filtered.empty:
            display_rank = filtered

    display_cols = [
        "順位",
        "テーマ",
        "流入ランク",
        "資金流入スコア",
        "流入タイプ",
        "流入理由",
        "ニュース材料",
        "関連ニュース件数",
        "平均5日上昇率%",
        "平均20日上昇率%",
        "平均出来高倍率",
        "20日高値3%以内数",
    ]
    display_cols = [c for c in display_cols if c in display_rank.columns]

    st.dataframe(display_rank[display_cols], use_container_width=True, hide_index=True)
    csv_rank = display_rank[display_cols].to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "テーマランキングCSV",
        data=csv_rank,
        file_name="theme_flow_ranking.csv",
        mime="text/csv",
        key=f"{tab_key}_rank_download",
    )




# =========================
# 値上がり率ランキング理由分析
# =========================

def get_latest_two_closes(price_df):
    if price_df is None or price_df.empty or len(price_df) < 2:
        return None
    df = price_df.dropna(subset=["Close"]).copy()
    if len(df) < 2:
        return None
    return df.iloc[-2], df.iloc[-1], df.index[-1]


def analyze_price_action_reason(price_df, code, name, market, sector33, sector17, theme):
    """
    値上がり率ランキング用。
    価格・出来高・高値更新・ローソク足から、上昇理由の仮説を作る。
    """
    if price_df is None or price_df.empty or len(price_df) < 30:
        return None

    df = price_df.copy()
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    prev = df.iloc[-2]
    last = df.iloc[-1]
    latest_date = df.index[-1]

    prev_close = float(prev["Close"])
    latest_open = float(last["Open"])
    latest_high = float(last["High"])
    latest_low = float(last["Low"])
    latest_close = float(last["Close"])
    latest_volume = float(last["Volume"])

    if prev_close <= 0 or latest_close <= 0:
        return None

    day_change_pct = (latest_close / prev_close - 1) * 100
    gap_pct = (latest_open / prev_close - 1) * 100 if latest_open > 0 else np.nan

    vol20 = volume.iloc[-21:-1].mean() if len(volume) >= 21 else np.nan
    vol5 = volume.iloc[-6:-1].mean() if len(volume) >= 6 else np.nan
    volume_ratio20 = latest_volume / vol20 if pd.notna(vol20) and vol20 > 0 else np.nan
    volume_ratio5 = latest_volume / vol5 if pd.notna(vol5) and vol5 > 0 else np.nan

    ret_3d = (latest_close / close.iloc[-4] - 1) * 100 if len(close) >= 4 and close.iloc[-4] > 0 else np.nan
    ret_5d = (latest_close / close.iloc[-6] - 1) * 100 if len(close) >= 6 and close.iloc[-6] > 0 else np.nan
    ret_20d = (latest_close / close.iloc[-21] - 1) * 100 if len(close) >= 21 and close.iloc[-21] > 0 else np.nan

    high20_prev = high.iloc[-21:-1].max() if len(high) >= 21 else np.nan
    high60_prev = high.iloc[-61:-1].max() if len(high) >= 61 else np.nan
    high20_update = pd.notna(high20_prev) and latest_close >= high20_prev
    high60_update = pd.notna(high60_prev) and latest_close >= high60_prev
    high20_distance_pct = (high20_prev / latest_close - 1) * 100 if pd.notna(high20_prev) and latest_close > 0 else np.nan

    ma25 = close.rolling(25).mean().iloc[-1]
    ma75 = close.rolling(75).mean().iloc[-1] if len(close) >= 75 else np.nan
    above_ma25 = pd.notna(ma25) and latest_close > ma25
    above_ma75 = pd.notna(ma75) and latest_close > ma75
    ma25_dev = (latest_close / ma25 - 1) * 100 if pd.notna(ma25) and ma25 > 0 else np.nan

    close_position = (latest_close - latest_low) / (latest_high - latest_low) if latest_high > latest_low else 0.5
    intraday_range_pct = (latest_high / latest_low - 1) * 100 if latest_low > 0 else np.nan
    upper_wick_pct = (latest_high - latest_close) / (latest_high - latest_low) * 100 if latest_high > latest_low else 0
    bullish = latest_close > latest_open

    reasons = []
    reason_tags = []

    if day_change_pct >= 15:
        reasons.append(f"前日比+{day_change_pct:.1f}%の急騰で短期資金が集中")
        reason_tags.append("急騰")
    elif day_change_pct >= 8:
        reasons.append(f"前日比+{day_change_pct:.1f}%の大幅高")
        reason_tags.append("大幅高")
    elif day_change_pct >= 4:
        reasons.append(f"前日比+{day_change_pct:.1f}%の強い上昇")
        reason_tags.append("上昇")

    if pd.notna(gap_pct) and gap_pct >= 5:
        reasons.append(f"寄り付きから+{gap_pct:.1f}%のGUで材料・需給反応が強い")
        reason_tags.append("GU")

    if pd.notna(volume_ratio20) and volume_ratio20 >= 5:
        reasons.append(f"出来高が20日平均比{volume_ratio20:.1f}倍に急増")
        reason_tags.append("出来高急増")
    elif pd.notna(volume_ratio20) and volume_ratio20 >= 2:
        reasons.append(f"出来高が20日平均比{volume_ratio20:.1f}倍で資金流入感あり")
        reason_tags.append("出来高増")

    if high60_update:
        reasons.append("60日高値を更新し、中期の上値抵抗を突破")
        reason_tags.append("60日高値更新")
    elif high20_update:
        reasons.append("20日高値を更新し、短期ブレイクが発生")
        reason_tags.append("20日高値更新")
    elif pd.notna(high20_distance_pct) and high20_distance_pct <= 2:
        reasons.append("20日高値付近まで上昇し、高値圏に接近")
        reason_tags.append("高値圏")

    if close_position >= 0.80:
        reasons.append("終値が当日レンジ上位で、高値引けに近い")
        reason_tags.append("高値引け")
    elif upper_wick_pct >= 45:
        reasons.append("上ヒゲがやや長く、短期利確も出ている")
        reason_tags.append("上ヒゲ注意")

    if pd.notna(ret_5d) and ret_5d >= 15:
        reasons.append(f"5日上昇率+{ret_5d:.1f}%でモメンタムが強い")
        reason_tags.append("短期モメンタム")

    if above_ma25 and above_ma75:
        reasons.append("25日線・75日線より上でトレンドが崩れていない")
        reason_tags.append("トレンド良好")
    elif above_ma25:
        reasons.append("25日線より上に回復し短期トレンドが改善")
        reason_tags.append("25日線回復")

    if pd.notna(ma25_dev) and ma25_dev >= 25:
        reasons.append(f"25日線乖離+{ma25_dev:.1f}%で短期過熱には注意")
        reason_tags.append("過熱注意")

    if not reasons:
        reasons.append("価格・出来高だけでは明確な理由を特定しづらい。開示・ニュース確認が必要")

    return {
        "コード": code,
        "銘柄名": name,
        "市場": market,
        "33業種": sector33,
        "17業種": sector17,
        "テーマ": theme,
        "最新日": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
        "終値": round(latest_close, 1),
        "前日終値": round(prev_close, 1),
        "値上がり率%": round(day_change_pct, 2),
        "GU率%": round(gap_pct, 2) if pd.notna(gap_pct) else np.nan,
        "出来高倍率20日": round(volume_ratio20, 2) if pd.notna(volume_ratio20) else np.nan,
        "出来高倍率5日": round(volume_ratio5, 2) if pd.notna(volume_ratio5) else np.nan,
        "3日上昇率%": round(ret_3d, 2) if pd.notna(ret_3d) else np.nan,
        "5日上昇率%": round(ret_5d, 2) if pd.notna(ret_5d) else np.nan,
        "20日上昇率%": round(ret_20d, 2) if pd.notna(ret_20d) else np.nan,
        "20日高値更新": bool(high20_update),
        "60日高値更新": bool(high60_update),
        "20日高値距離%": round(high20_distance_pct, 2) if pd.notna(high20_distance_pct) else np.nan,
        "終値位置": round(close_position, 2),
        "日中値幅%": round(intraday_range_pct, 2) if pd.notna(intraday_range_pct) else np.nan,
        "上ヒゲ%": round(upper_wick_pct, 1),
        "25日線乖離%": round(ma25_dev, 2) if pd.notna(ma25_dev) else np.nan,
        "価格出来高理由": " / ".join(reasons[:6]),
        "理由タグ": " / ".join(dict.fromkeys(reason_tags)),
        "Yahooチャート": f"https://finance.yahoo.co.jp/quote/{code}.T/chart",
    }


def build_disclosure_map_for_dates(target_date, lookback_days=2, max_items_per_day=500):
    """指定日から過去数日分のTDnetタイトルをコード別にまとめる。"""
    disclosure_map = {}
    errors = []

    for i in range(int(lookback_days)):
        d = target_date - timedelta(days=i)
        date_text = d.strftime("%Y-%m-%d")
        try:
            df, error_msg = fetch_tdnet_disclosures_free(date_text, max_items=int(max_items_per_day))
            if error_msg:
                errors.append(error_msg)
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                code = str(row.get("コード", "")).strip().upper().replace(".T", "")
                if not code:
                    continue
                title = str(row.get("タイトル", "")).strip()
                time_text = str(row.get("時刻", "")).strip()
                pdf_link = str(row.get("PDFリンク", "")).strip()
                company = str(row.get("銘柄名", "")).strip()
                disclosure_map.setdefault(code, []).append({
                    "date": date_text,
                    "time": time_text,
                    "title": title,
                    "company": company,
                    "pdf_link": pdf_link,
                })
        except Exception as e:
            errors.append(f"{date_text}: {e}")

    return disclosure_map, errors


def summarize_disclosure_reason(code, disclosure_map):
    items = disclosure_map.get(str(code).strip().upper().replace(".T", ""), [])
    if not items:
        return "", "", ""

    titles = []
    material_tags = []
    pdf_links = []
    for item in items[:4]:
        title = item.get("title", "")
        if not title:
            continue
        label = f"{item.get('date', '')} {item.get('time', '')} {title}".strip()
        titles.append(label)
        pdf = item.get("pdf_link", "")
        if pdf:
            pdf_links.append(pdf)

        # タイトルだけでも最低限の材料タイプを拾う
        judgement = judge_disclosure_material(title=title, body_text="", code=code, company=item.get("company", ""))
        good = str(judgement.get("好材料ポイント", "")).strip()
        bad = str(judgement.get("悪材料ポイント", "")).strip()
        label2 = str(judgement.get("材料判定", "")).strip()
        if good:
            material_tags.append(good)
        if bad:
            material_tags.append(bad)
        if label2 and label2 not in ["中立", "要確認"]:
            material_tags.append(label2)

    reason = " / ".join(titles[:3])
    tags = " / ".join(dict.fromkeys([x for x in material_tags if x]))
    pdf_link = pdf_links[0] if pdf_links else ""
    return reason, tags, pdf_link


def build_company_news_queries(code, name, days=3):
    d = int(days)
    code = str(code).replace(".T", "")
    name = str(name).strip()
    return [
        f'"{name}" 株 急騰 上昇 材料 when:{d}d',
        f'{code} {name} 決算 上方修正 増配 受注 提携 when:{d}d',
    ]


def summarize_company_news_reason(code, name, news_days=3, max_items=3):
    articles = []
    for q in build_company_news_queries(code, name, days=news_days):
        articles.extend(fetch_google_news_rss(q, max_items=max_items))
    articles = dedupe_news_articles(articles)

    if not articles:
        return "ネットニュースで明確な個別材料は取得できず", "", ""

    titles = [str(a.get("title", "")).strip() for a in articles if str(a.get("title", "")).strip()]
    keywords = extract_news_keywords(articles)

    title_text = " / ".join(titles[:2])
    if keywords:
        reason = f"ニュース見出しで「{'、'.join(keywords[:5])}」を確認。関連材料：{title_text}"
    else:
        reason = f"関連ニュース見出しを確認：{title_text}"

    evidence = []
    for a in articles[:3]:
        title = a.get("title", "")
        source = a.get("source", "")
        published = a.get("published", "")
        if title:
            evidence.append(f"{published} {source}: {title}".strip())

    first_link = articles[0].get("link", "") if articles else ""
    return reason, " / ".join(evidence), first_link


def combine_gainer_reasons(price_reason, disclosure_reason, disclosure_tags, news_reason):
    parts = []
    if disclosure_reason:
        if disclosure_tags:
            parts.append(f"開示材料：{disclosure_tags}")
        else:
            parts.append("開示材料あり")
    if news_reason and "取得できず" not in news_reason:
        parts.append(news_reason)
    if price_reason:
        parts.append(f"値動き：{price_reason}")
    if not parts:
        return "明確な材料は未確認。値動き・出来高主導の可能性。"
    return " / ".join(parts[:3])


def run_top_gainers_reason_screening(
    market_choice,
    max_scan,
    wait_sec,
    target_date,
    disclosure_lookback_days,
    use_tdnet,
    use_news,
    news_days,
    news_top_n,
):
    try:
        all_stocks = load_jpx_list()
    except Exception as e:
        st.error(f"JPXの銘柄リスト取得に失敗しました: {e}")
        st.stop()

    target_stocks = filter_market(all_stocks, market_choice)
    if target_stocks.empty:
        st.warning("対象銘柄が見つかりませんでした。")
        st.stop()

    total_all = len(target_stocks)
    if max_scan > 0:
        target_stocks = target_stocks.head(int(max_scan))
    total = len(target_stocks)

    st.write(f"対象市場：{market_choice}")
    st.write(f"確認銘柄数：{total} / 選択市場全体 {total_all}")

    disclosure_map = {}
    disclosure_errors = []
    if use_tdnet:
        with st.spinner("TDnet開示タイトルを確認中..."):
            disclosure_map, disclosure_errors = build_disclosure_map_for_dates(
                target_date=target_date,
                lookback_days=int(disclosure_lookback_days),
                max_items_per_day=700,
            )
        if disclosure_errors:
            st.caption("TDnet取得で一部エラーあり。取得できた範囲で続行します。")

    progress_bar = st.progress(0)
    status_area = st.empty()
    results = []
    failed_count = 0

    for n, (_, row) in enumerate(target_stocks.iterrows(), start=1):
        code = row["コード"]
        name = row["銘柄名"]
        market = row["市場"]
        sector33 = row.get("33業種", "")
        sector17 = row.get("17業種", "")
        theme = row.get("テーマ", infer_theme(code, name, sector33, sector17))

        status_area.write(f"値上がり率確認中：{n}/{total} {code} {name}")

        try:
            price_df = download_price(code)
            result = analyze_price_action_reason(
                price_df=price_df,
                code=code,
                name=name,
                market=market,
                sector33=sector33,
                sector17=sector17,
                theme=theme,
            )
            if result is not None:
                results.append(result)
        except Exception:
            failed_count += 1

        progress_bar.progress(n / total)
        if wait_sec > 0:
            time.sleep(wait_sec)

    status_area.write("値上がり率ランキング作成中...")

    if not results:
        st.session_state["top_gainers_reason_df"] = pd.DataFrame()
        st.session_state["top_gainers_failed_count"] = failed_count
        return

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("値上がり率%", ascending=False).head(50).reset_index(drop=True)
    result_df.insert(0, "順位", range(1, len(result_df) + 1))

    # トップ50になったものだけ、開示・ニュース理由を付ける。
    rows = []
    news_n = min(int(news_top_n), len(result_df))
    for idx, row in result_df.iterrows():
        code = row.get("コード", "")
        name = row.get("銘柄名", "")

        disclosure_reason = ""
        disclosure_tags = ""
        pdf_link = ""
        if use_tdnet:
            disclosure_reason, disclosure_tags, pdf_link = summarize_disclosure_reason(code, disclosure_map)

        news_reason = ""
        news_evidence = ""
        news_link = ""
        if use_news and idx < news_n:
            status_area.write(f"ニュース理由確認中：{idx + 1}/{news_n} {code} {name}")
            news_reason, news_evidence, news_link = summarize_company_news_reason(
                code=code,
                name=name,
                news_days=int(news_days),
                max_items=3,
            )

        combined = combine_gainer_reasons(
            price_reason=row.get("価格出来高理由", ""),
            disclosure_reason=disclosure_reason,
            disclosure_tags=disclosure_tags,
            news_reason=news_reason,
        )

        out = row.to_dict()
        out["推定上昇理由"] = combined
        out["TDnet開示理由"] = disclosure_reason if disclosure_reason else "該当開示なし"
        out["TDnet材料タグ"] = disclosure_tags
        out["ニュース理由"] = news_reason
        out["参考ニュース"] = news_evidence
        out["ニュースリンク"] = news_link
        out["PDFリンク"] = pdf_link
        rows.append(out)

    final_df = pd.DataFrame(rows)
    st.session_state["top_gainers_reason_df"] = final_df
    st.session_state["top_gainers_failed_count"] = failed_count
    status_area.write("値上がり率ランキング理由分析完了")


def render_top_gainers_reason_result():
    result_df = st.session_state.get("top_gainers_reason_df", pd.DataFrame())
    failed_count = st.session_state.get("top_gainers_failed_count", 0)

    st.subheader("値上がり率ランキング Top50・理由分析")

    if result_df is None or result_df.empty:
        st.warning("結果がありません。市場や確認銘柄数を変えて再実行してください。")
        return

    st.write(f"表示件数：{len(result_df)}")
    st.write(f"取得失敗数：{failed_count}")

    display_cols = [
        "順位", "コード", "銘柄名", "市場", "テーマ", "最新日", "終値", "値上がり率%",
        "推定上昇理由", "TDnet開示理由", "ニュース理由", "価格出来高理由",
        "出来高倍率20日", "GU率%", "5日上昇率%", "20日上昇率%",
        "20日高値更新", "60日高値更新", "終値位置", "上ヒゲ%", "Yahooチャート",
    ]
    display_cols = [c for c in display_cols if c in result_df.columns]

    st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)

    csv = result_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "CSVダウンロード",
        data=csv,
        file_name="top50_gainers_reason.csv",
        mime="text/csv",
        key="top_gainers_reason_download",
    )

    st.subheader("詳細確認")
    choices = (
        result_df["順位"].astype(str) + "位｜" + result_df["コード"].astype(str) + " "
        + result_df["銘柄名"].astype(str) + "｜+" + result_df["値上がり率%"].astype(str) + "%"
    ).tolist()
    selected = st.selectbox("詳細を見る銘柄", choices, key="top_gainers_detail_select")
    pos = choices.index(selected)
    row = result_df.iloc[pos]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("順位", f"{row.get('順位', '-') }位")
    c2.metric("値上がり率", f"{row.get('値上がり率%', '-') }%")
    c3.metric("出来高倍率20日", row.get("出来高倍率20日", "-"))
    c4.metric("テーマ", row.get("テーマ", "-"))

    st.write("### 推定上昇理由")
    st.write(row.get("推定上昇理由", ""))

    st.write("### TDnet開示")
    st.write(row.get("TDnet開示理由", ""))
    if row.get("PDFリンク", ""):
        st.write(row.get("PDFリンク", ""))

    st.write("### 参考ニュース")
    st.write(row.get("参考ニュース", ""))
    if row.get("ニュースリンク", ""):
        st.write(row.get("ニュースリンク", ""))

    detail_cols = [
        "コード", "銘柄名", "市場", "33業種", "17業種", "テーマ", "最新日", "終値", "前日終値",
        "値上がり率%", "GU率%", "出来高倍率20日", "出来高倍率5日", "3日上昇率%",
        "5日上昇率%", "20日上昇率%", "20日高値更新", "60日高値更新", "20日高値距離%",
        "終値位置", "日中値幅%", "上ヒゲ%", "25日線乖離%", "理由タグ", "Yahooチャート",
    ]
    detail_cols = [c for c in detail_cols if c in result_df.columns]
    st.write("### 詳細データ")
    st.dataframe(pd.DataFrame(row[detail_cols]).rename(columns={row.name: "値"}), use_container_width=True)


def render_top_gainers_reason_tab(tab_key="top_gainers_reason"):
    st.subheader("値上がり率ランキング Top50・理由分析")
    st.write(
        "選択市場の銘柄を確認し、最新取引日の値上がり率トップ50を作ります。"
        "各銘柄について、価格・出来高・高値更新・TDnet開示・ネットニュースから上昇理由を推定します。"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        target_date = st.date_input(
            "TDnet開示の確認基準日",
            value=datetime.now(timezone(timedelta(hours=9))).date(),
            key=f"{tab_key}_target_date",
        )
        disclosure_lookback_days = st.selectbox(
            "TDnet開示を何日分見るか",
            [1, 2, 3, 5],
            index=2,
            key=f"{tab_key}_disclosure_days",
        )
    with col2:
        use_tdnet = st.checkbox("TDnet開示タイトルも理由に使う", value=True, key=f"{tab_key}_use_tdnet")
        use_news = st.checkbox("ネットニュースも理由に使う", value=True, key=f"{tab_key}_use_news")
    with col3:
        news_days = st.selectbox("ニュース検索期間", [1, 3, 7, 14], index=1, key=f"{tab_key}_news_days")
        news_top_n = st.slider(
            "ニュース検索する上位件数",
            min_value=0,
            max_value=50,
            value=50,
            step=5,
            key=f"{tab_key}_news_top_n",
            help="50にするとトップ50すべてニュース検索します。重い場合は20に下げてください。",
        )

    st.info("全市場・ニュース50件検索は時間がかかります。最初は確認銘柄数100〜300で動作確認してください。")

    if st.button("値上がり率Top50の理由を分析", type="primary", key=f"{tab_key}_run"):
        run_top_gainers_reason_screening(
            market_choice=market_choice,
            max_scan=max_scan,
            wait_sec=wait_sec,
            target_date=target_date,
            disclosure_lookback_days=disclosure_lookback_days,
            use_tdnet=use_tdnet,
            use_news=use_news,
            news_days=news_days,
            news_top_n=news_top_n,
        )

    if "top_gainers_reason_df" in st.session_state:
        render_top_gainers_reason_result()





def resample_to_weekly_ohlcv(daily_df):
    """日足OHLCVを金曜終値ベースの週足へ変換する。"""
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in needed):
        return pd.DataFrame()

    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        return pd.DataFrame()

    weekly = df[needed].resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    return weekly.dropna(subset=["Open", "High", "Low", "Close"])

# ============================================================
# ウェッジブレイク（WB）/ ウェッジブレイク・プルバック（WBPB）
# ============================================================
def fit_falling_wedge(pattern_df, params):
    """
    下降ウェッジを一次回帰で近似する。
    高値側の下降傾きが安値側より急で、値幅が収束している形を探す。
    """
    if pattern_df is None or pattern_df.empty or len(pattern_df) < 12:
        return None

    high = pattern_df["High"].astype(float).reset_index(drop=True)
    low = pattern_df["Low"].astype(float).reset_index(drop=True)
    close = pattern_df["Close"].astype(float).reset_index(drop=True)
    volume = pattern_df["Volume"].astype(float).reset_index(drop=True)

    n = len(pattern_df)
    x = np.arange(n, dtype=float)

    try:
        high_slope, high_intercept = np.polyfit(x, high, 1)
        low_slope, low_intercept = np.polyfit(x, low, 1)
    except Exception:
        return None

    base_price = float(close.mean())
    if base_price <= 0:
        return None

    upper_start = high_intercept
    lower_start = low_intercept
    upper_end = high_slope * (n - 1) + high_intercept
    lower_end = low_slope * (n - 1) + low_intercept

    start_width = upper_start - lower_start
    end_width = upper_end - lower_end

    if start_width <= 0 or end_width <= 0:
        return None

    convergence_ratio = end_width / start_width
    high_slope_pct = high_slope / base_price * 100
    low_slope_pct = low_slope / base_price * 100

    upper_fit = high_slope * x + high_intercept
    lower_fit = low_slope * x + low_intercept
    tolerance = base_price * params["touch_tolerance_pct"] / 100

    upper_touches = int((high >= (upper_fit - tolerance)).sum())
    lower_touches = int((low <= (lower_fit + tolerance)).sum())

    split = max(2, n // 2)
    first_volume = float(volume.iloc[:split].mean())
    second_volume = float(volume.iloc[split:].mean())
    volume_contract_ratio = second_volume / first_volume if first_volume > 0 else np.nan

    # 下降ウェッジ：
    # ・上側のトレンドラインは下降（またはゆるい横ばい）
    # ・上側の下げ方の方が下側より急で、値幅が収束
    # ・上下に複数回接触
    is_wedge = (
        high_slope_pct <= params["upper_slope_max_pct_per_day"]
        and high_slope < low_slope
        and convergence_ratio <= params["convergence_ratio_max"]
        and upper_touches >= params["min_upper_touches"]
        and lower_touches >= params["min_lower_touches"]
        and (
            pd.isna(volume_contract_ratio)
            or volume_contract_ratio <= params["volume_contract_ratio_max"]
        )
    )

    return {
        "is_wedge": bool(is_wedge),
        "n": n,
        "high_slope": float(high_slope),
        "low_slope": float(low_slope),
        "high_intercept": float(high_intercept),
        "low_intercept": float(low_intercept),
        "high_slope_pct": float(high_slope_pct),
        "low_slope_pct": float(low_slope_pct),
        "upper_start": float(upper_start),
        "lower_start": float(lower_start),
        "upper_end": float(upper_end),
        "lower_end": float(lower_end),
        "start_width": float(start_width),
        "end_width": float(end_width),
        "convergence_ratio": float(convergence_ratio),
        "upper_touches": upper_touches,
        "lower_touches": lower_touches,
        "volume_contract_ratio": float(volume_contract_ratio) if pd.notna(volume_contract_ratio) else np.nan,
    }


def wedge_line_value(fit, step):
    return fit["high_slope"] * step + fit["high_intercept"]


def wedge_lower_line_value(fit, step):
    return fit["low_slope"] * step + fit["low_intercept"]


def calc_wedge_targets(entry, stop_line, fit, prior_high):
    """
    ウェッジの始点の値幅を測定値幅として使い、利確候補を作る。
    """
    measure_move = max(float(fit["start_width"]), entry * 0.04)
    target_1 = max(float(prior_high), entry + measure_move * 0.55)
    target_2 = entry + measure_move

    risk = entry - stop_line
    loss_pct = risk / entry * 100 if entry > 0 else np.nan
    rr1 = (target_1 - entry) / risk if risk > 0 else np.nan
    rr2 = (target_2 - entry) / risk if risk > 0 else np.nan

    return target_1, target_2, loss_pct, rr1, rr2


def find_wedge_pullback_setup(df, params):
    """
    WBPBを探す。
    2〜pullback_max_days日前にウェッジを初回ブレイクした後、
    押しを挟み、今日が直近スイング高値を再上抜けした形を検出する。
    """
    total_len = len(df)
    lookback = params["lookback"]
    prior_days = params["prior_trend_days"]
    latest_idx = total_len - 1

    if total_len < lookback + prior_days + 10:
        return None

    max_days = min(params["pullback_max_days"], latest_idx - lookback - prior_days)
    if max_days < 2:
        return None

    best = None

    for days_since in range(2, max_days + 1):
        break_idx = latest_idx - days_since

        pattern_start = break_idx - lookback
        prior_start = pattern_start - prior_days

        if prior_start < 0:
            continue

        pattern_df = df.iloc[pattern_start:break_idx].copy()
        pre_df = df.iloc[prior_start:pattern_start].copy()

        fit = fit_falling_wedge(pattern_df, params)
        if fit is None or not fit["is_wedge"]:
            continue

        if pre_df.empty:
            continue

        prior_rise_pct = (
            (float(pattern_df["Close"].iloc[0]) / float(pre_df["Close"].iloc[0]) - 1) * 100
            if float(pre_df["Close"].iloc[0]) > 0
            else np.nan
        )
        if pd.isna(prior_rise_pct) or prior_rise_pct < params["min_prior_rise_pct"]:
            continue

        break_row = df.iloc[break_idx]
        break_close = float(break_row["Close"])
        break_open = float(break_row["Open"])
        break_high = float(break_row["High"])
        break_volume = float(break_row["Volume"])

        upper_break_line = wedge_line_value(fit, lookback)
        prev_vol = float(df["Volume"].iloc[max(0, break_idx - 5):break_idx].mean())
        break_volume_ratio = break_volume / prev_vol if prev_vol > 0 else np.nan
        break_close_pos = calc_close_position(break_row)

        is_initial_break = (
            break_close > upper_break_line * (1 + params["breakout_margin_pct"] / 100)
            and break_close > break_open
            and (pd.isna(break_volume_ratio) or break_volume_ratio >= params["breakout_volume_min"])
            and break_close_pos >= params["close_position_min"]
        )
        if not is_initial_break:
            continue

        # 初回ブレイク後〜昨日まで。最低1日の押しを要求。
        post_df = df.iloc[break_idx + 1:latest_idx].copy()
        if len(post_df) < 1:
            continue

        pivot = float(df["High"].iloc[break_idx:latest_idx].max())
        pullback_low = float(post_df["Low"].min())
        pullback_pct = (pivot - pullback_low) / pivot * 100 if pivot > 0 else np.nan

        latest = df.iloc[-1]
        latest_close = float(latest["Close"])
        latest_open = float(latest["Open"])
        latest_volume = float(latest["Volume"])
        latest_close_pos = calc_close_position(latest)
        latest_prev_close = float(df["Close"].iloc[-2])

        latest_prev5_volume = float(df["Volume"].iloc[-6:-1].mean()) if len(df) >= 6 else np.nan
        latest_volume_ratio = latest_volume / latest_prev5_volume if pd.notna(latest_prev5_volume) and latest_prev5_volume > 0 else np.nan

        # 今日が「押し後のスイング高値」を新たに抜いていること。
        is_rebreak = (
            latest_close > pivot * (1 + params["breakout_margin_pct"] / 100)
            and latest_prev_close <= pivot * (1 + params["breakout_margin_pct"] / 100)
            and latest_close > latest_open
            and latest_close_pos >= params["close_position_min"]
            and (pd.isna(latest_volume_ratio) or latest_volume_ratio >= params["pullback_volume_min"])
            and pd.notna(pullback_pct)
            and pullback_pct >= params["pullback_min_pct"]
        )

        if not is_rebreak:
            continue

        stop_line = pullback_low * 0.99
        prior_high = max(float(pattern_df["High"].max()), pivot)
        target_1, target_2, loss_pct, rr1, rr2 = calc_wedge_targets(
            latest_close, stop_line, fit, prior_high
        )

        candidate = {
            "fit": fit,
            "break_idx": break_idx,
            "days_since_breakout": days_since,
            "break_date": df.index[break_idx],
            "breakout_line": pivot,
            "stop_line": stop_line,
            "target_1": target_1,
            "target_2": target_2,
            "loss_pct": loss_pct,
            "rr1": rr1,
            "rr2": rr2,
            "volume_ratio": latest_volume_ratio,
            "pullback_pct": pullback_pct,
            "prior_rise_pct": prior_rise_pct,
        }

        # 直近の初回ブレイクを優先
        if best is None or candidate["days_since_breakout"] < best["days_since_breakout"]:
            best = candidate

    return best


def analyze_wedge_break(
    price_df,
    code,
    name,
    market,
    params,
    use_ma200_filter=False,
    timeframe_label="日足",
    ma_unit="日"
):
    """
    WB / WBPBを日足ベースで判定する。
    完全な裁量チャート認識ではなく、上昇トレンド中の下降ウェッジを
    数値条件に置き換えたスクリーニングです。
    """
    lookback = params["lookback"]
    prior_days = params["prior_trend_days"]

    if price_df is None or price_df.empty or len(price_df) < lookback + prior_days + 80:
        return None

    df = price_df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    df["MA25"] = df["Close"].rolling(25).mean()
    df["MA75"] = df["Close"].rolling(75).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    latest = df.iloc[-1]
    latest_date = df.index[-1]
    latest_close = float(latest["Close"])
    latest_open = float(latest["Open"])
    latest_high = float(latest["High"])
    latest_low = float(latest["Low"])
    latest_volume = float(latest["Volume"])
    latest_close_pos = calc_close_position(latest)

    ma25 = float(df["MA25"].iloc[-1]) if pd.notna(df["MA25"].iloc[-1]) else np.nan
    ma75 = float(df["MA75"].iloc[-1]) if pd.notna(df["MA75"].iloc[-1]) else np.nan
    ma200 = float(df["MA200"].iloc[-1]) if pd.notna(df["MA200"].iloc[-1]) else np.nan

    if pd.isna(ma25) or pd.isna(ma75):
        return None

    if use_ma200_filter and (pd.isna(ma200) or latest_close < ma200):
        return None

    ma75_up = (
        pd.notna(df["MA75"].iloc[-6])
        and float(df["MA75"].iloc[-1]) > float(df["MA75"].iloc[-6])
    )
    if params["need_ma75_up"] and not ma75_up:
        return None

    # 今日を除いた直近期間をウェッジとして判定する
    pattern_start = len(df) - lookback - 1
    pattern_end = len(df) - 1
    prior_start = pattern_start - prior_days

    if prior_start < 0:
        return None

    pattern_df = df.iloc[pattern_start:pattern_end].copy()
    pre_df = df.iloc[prior_start:pattern_start].copy()

    fit = fit_falling_wedge(pattern_df, params)
    if fit is None:
        return None

    prior_rise_pct = (
        (float(pattern_df["Close"].iloc[0]) / float(pre_df["Close"].iloc[0]) - 1) * 100
        if not pre_df.empty and float(pre_df["Close"].iloc[0]) > 0
        else np.nan
    )

    upper_today = wedge_line_value(fit, lookback)
    lower_today = wedge_lower_line_value(fit, lookback)
    distance_to_upper_pct = (
        (latest_close / upper_today - 1) * 100
        if upper_today > 0
        else np.nan
    )

    prev5_volume = float(df["Volume"].iloc[-6:-1].mean()) if len(df) >= 6 else np.nan
    volume_ratio = latest_volume / prev5_volume if pd.notna(prev5_volume) and prev5_volume > 0 else np.nan

    direct_break = (
        fit["is_wedge"]
        and pd.notna(prior_rise_pct)
        and prior_rise_pct >= params["min_prior_rise_pct"]
        and latest_close > upper_today * (1 + params["breakout_margin_pct"] / 100)
        and latest_close > latest_open
        and latest_close_pos >= params["close_position_min"]
        and (pd.isna(volume_ratio) or volume_ratio >= params["breakout_volume_min"])
    )

    # WBPBは、WBの初回ブレイク後に押しを作って再度スイング高値を抜く形
    pullback_setup = find_wedge_pullback_setup(df, params)

    if direct_break:
        setup_type = "WB"
        entry_line = upper_today
        stop_line = min(lower_today * 0.99, float(df["Low"].tail(5).min()) * 0.99)
        prior_high = max(float(pattern_df["High"].max()), float(pre_df["High"].max()))
        target_1, target_2, loss_pct, rr1, rr2 = calc_wedge_targets(
            latest_close, stop_line, fit, prior_high
        )
        pullback_pct = np.nan
        first_break_date = latest_date
        comment = "下降ウェッジの上限（DTL）を出来高を伴って上抜ける初動。"
        entry_method = "下降トレンドライン（DTL）を終値で上抜け"
    elif pullback_setup is not None:
        setup_type = "WBPB"
        entry_line = pullback_setup["breakout_line"]
        stop_line = pullback_setup["stop_line"]
        target_1 = pullback_setup["target_1"]
        target_2 = pullback_setup["target_2"]
        loss_pct = pullback_setup["loss_pct"]
        rr1 = pullback_setup["rr1"]
        rr2 = pullback_setup["rr2"]
        volume_ratio = pullback_setup["volume_ratio"]
        pullback_pct = pullback_setup["pullback_pct"]
        first_break_date = pullback_setup["break_date"]
        comment = "初回ウェッジブレイク後の押しを経て、スイング高値を再度上抜ける形。"
        entry_method = "押し後のスイング高値を終値で上抜け"
    else:
        # ブレイク前の監視候補
        is_monitor = (
            fit["is_wedge"]
            and pd.notna(prior_rise_pct)
            and prior_rise_pct >= params["min_prior_rise_pct"]
            and pd.notna(distance_to_upper_pct)
            and -params["monitor_distance_pct"] <= distance_to_upper_pct <= params["breakout_margin_pct"]
            and latest_close >= lower_today
        )
        if not is_monitor:
            return None

        setup_type = "WB監視"
        entry_line = upper_today
        stop_line = min(lower_today * 0.99, float(df["Low"].tail(5).min()) * 0.99)
        prior_high = max(float(pattern_df["High"].max()), float(pre_df["High"].max()))
        target_1, target_2, loss_pct, rr1, rr2 = calc_wedge_targets(
            max(entry_line, latest_close), stop_line, fit, prior_high
        )
        pullback_pct = np.nan
        first_break_date = pd.NaT
        comment = "下降ウェッジを形成中。上限の下降トレンドライン上抜けを待つ。"
        entry_method = "DTL上抜け待ち"

    risk_ok = pd.notna(loss_pct) and loss_pct > 0 and loss_pct <= params["stop_loss_max_pct"]
    rr_ok = pd.notna(rr2) and rr2 >= params["rr_min"]

    if setup_type in ["WB", "WBPB"] and risk_ok and rr_ok:
        buy_judge = "候補"
        priority = "A" if setup_type == "WBPB" or (pd.notna(volume_ratio) and volume_ratio >= params["breakout_volume_min"] * 1.4) else "B"
        skip_reason = "なし"
    elif setup_type in ["WB", "WBPB"]:
        buy_judge = "監視"
        priority = "B"
        reasons = []
        if not risk_ok:
            reasons.append("損切り幅が広い")
        if not rr_ok:
            reasons.append("RR不足")
        skip_reason = " / ".join(reasons) if reasons else "ブレイクは確認、条件の最終確認待ち"
    else:
        buy_judge = "監視"
        priority = "B"
        skip_reason = "出来高を伴うDTL上抜け待ち"

    # スコア：優先順位付け用
    score = 0.0
    score += min(max(prior_rise_pct, 0), 25) * 1.2 if pd.notna(prior_rise_pct) else 0
    score += max(0, 1.0 - fit["convergence_ratio"]) * 35
    score += min(fit["upper_touches"], 4) * 3
    score += min(fit["lower_touches"], 4) * 3
    score += min(volume_ratio, 4) * 5 if pd.notna(volume_ratio) else 0
    score += min(rr2, 4) * 6 if pd.notna(rr2) else 0
    if setup_type == "WB":
        score += 18
    elif setup_type == "WBPB":
        score += 24
    else:
        score += 8

    return {
        "コード": code,
        "銘柄名": name,
        "市場": market,
        "足種": timeframe_label,
        "セットアップ": setup_type,
        "判定": "ウェッジブレイク",
        "買い候補": buy_judge,
        "優先度": priority,
        "エントリー方法": entry_method,
        "上抜けライン": round(float(entry_line), 1),
        "損切りライン": round(float(stop_line), 1),
        "第1利確ライン": round(float(target_1), 1),
        "第2利確ライン": round(float(target_2), 1),
        "想定損失%": round(float(loss_pct), 2) if pd.notna(loss_pct) else np.nan,
        "第1RR": round(float(rr1), 2) if pd.notna(rr1) else np.nan,
        "第2RR": round(float(rr2), 2) if pd.notna(rr2) else np.nan,
        "見送り理由": skip_reason,
        "コメント": comment,
        "最新日": latest_date.strftime("%Y-%m-%d"),
        "終値": round(latest_close, 1),
        f"25{ma_unit}線": round(ma25, 1) if pd.notna(ma25) else np.nan,
        f"75{ma_unit}線": round(ma75, 1) if pd.notna(ma75) else np.nan,
        f"200{ma_unit}線": round(ma200, 1) if pd.notna(ma200) else np.nan,
        f"25{ma_unit}線上": "はい" if latest_close >= ma25 else "いいえ",
        f"75{ma_unit}線上": "はい" if latest_close >= ma75 else "いいえ",
        f"75{ma_unit}線上向き": "はい" if ma75_up else "いいえ",
        "上昇トレンド前段%": round(float(prior_rise_pct), 2) if pd.notna(prior_rise_pct) else np.nan,
        "ウェッジ期間": int(lookback),
        "値幅収束率": round(float(fit["convergence_ratio"]), 2),
        "上限傾き%/日": round(float(fit["high_slope_pct"]), 3),
        "下限傾き%/日": round(float(fit["low_slope_pct"]), 3),
        "上限接触数": int(fit["upper_touches"]),
        "下限接触数": int(fit["lower_touches"]),
        "保ち合い出来高収縮率": round(float(fit["volume_contract_ratio"]), 2) if pd.notna(fit["volume_contract_ratio"]) else np.nan,
        "出来高倍率": round(float(volume_ratio), 2) if pd.notna(volume_ratio) else np.nan,
        "終値位置": round(float(latest_close_pos), 2),
        "上限ラインまで%": round(float(distance_to_upper_pct), 2) if pd.notna(distance_to_upper_pct) else np.nan,
        "押し率%": round(float(pullback_pct), 2) if pd.notna(pullback_pct) else np.nan,
        "初回ブレイク日": first_break_date.strftime("%Y-%m-%d") if pd.notna(first_break_date) else "",
        "スコア": round(float(score), 2),
        "Yahooチャート": f"https://finance.yahoo.co.jp/quote/{code}.T/chart",
    }


def show_saved_wedge_results():
    result_key = "screen_result_df_wedge"
    failed_key = "screen_failed_count_wedge"
    market_key = "screen_market_wedge"

    if result_key not in st.session_state:
        return

    result_df = st.session_state[result_key]
    failed_count = st.session_state.get(failed_key, 0)
    saved_market = st.session_state.get(market_key, "")

    st.subheader("抽出結果")

    if result_df is None or result_df.empty:
        st.warning("条件に一致する銘柄は見つかりませんでした。判定を「普通」または「ゆるめ」にするか、確認銘柄数を増やしてください。")
        return

    st.write(f"抽出銘柄数：{len(result_df)}")
    st.write(f"取得失敗数：{failed_count}")

    col1, col2, col3 = st.columns(3)
    with col1:
        judge_filter = st.selectbox(
            "表示する買い候補",
            ["すべて", "候補のみ", "候補・監視", "監視のみ"],
            key="judge_filter_wedge"
        )
    with col2:
        setup_filter = st.multiselect(
            "表示するセットアップ",
            ["WB", "WBPB", "WB監視"],
            default=["WB", "WBPB", "WB監視"],
            key="setup_filter_wedge"
        )
    with col3:
        timeframe_options = [x for x in ["日足", "週足"] if x in result_df.get("足種", pd.Series(dtype=str)).astype(str).unique().tolist()]
        timeframe_filter = st.multiselect(
            "表示する足種",
            timeframe_options,
            default=timeframe_options,
            key="timeframe_filter_wedge"
        )

    display_df = result_df.copy()

    if judge_filter == "候補のみ":
        display_df = display_df[display_df["買い候補"] == "候補"]
    elif judge_filter == "候補・監視":
        display_df = display_df[display_df["買い候補"].isin(["候補", "監視"])]
    elif judge_filter == "監視のみ":
        display_df = display_df[display_df["買い候補"] == "監視"]

    if setup_filter:
        display_df = display_df[display_df["セットアップ"].isin(setup_filter)]
    if timeframe_filter:
        display_df = display_df[display_df["足種"].isin(timeframe_filter)]

    display_df = display_df.drop(columns=["_買い候補順", "_優先度順", "_セットアップ順", "_足種順"], errors="ignore")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = display_df.to_csv(index=False, encoding="utf-8-sig")
    market_for_filename = str(saved_market).replace("全市場", "all_markets")
    st.download_button(
        label="CSVダウンロード",
        data=csv,
        file_name=f"wedge_break_{market_for_filename}_trade_plan.csv",
        mime="text/csv",
        key=f"download_wedge_{market_for_filename}"
    )


def run_wedge_screener(
    market_choice,
    preset_name,
    timeframe_choice,
    max_scan,
    wait_sec,
    use_ma200_filter
):
    try:
        all_stocks = load_jpx_list()
    except Exception as e:
        st.error(f"JPXの銘柄リスト取得に失敗しました: {e}")
        st.stop()

    target_stocks = filter_market(all_stocks, market_choice)
    if target_stocks.empty:
        st.warning("対象銘柄が見つかりませんでした。")
        st.stop()

    total_all = len(target_stocks)
    if max_scan > 0:
        target_stocks = target_stocks.head(max_scan)

    total = len(target_stocks)
    st.write(f"対象市場：{market_choice}")
    st.write(f"対象銘柄数：{total} / 選択市場全体 {total_all}")

    scan_daily = timeframe_choice in ["日足", "日足・週足の両方"]
    scan_weekly = timeframe_choice in ["週足", "日足・週足の両方"]

    progress_bar = st.progress(0)
    status_area = st.empty()
    results = []
    failed_count = 0

    for n, (_, row) in enumerate(target_stocks.iterrows(), start=1):
        code = row["コード"]
        name = row["銘柄名"]
        market = row["市場"]

        status_area.write(f"{n}/{total} 確認中：{code} {name}")

        try:
            if scan_daily:
                daily_params = WEDGE_PRESETS[preset_name]
                daily_df = download_price(code)
                result = analyze_wedge_break(
                    price_df=daily_df,
                    code=code,
                    name=name,
                    market=market,
                    params=daily_params,
                    use_ma200_filter=use_ma200_filter,
                    timeframe_label="日足",
                    ma_unit="日"
                )
                if result is not None:
                    result["33業種"] = row.get("33業種", "")
                    result["17業種"] = row.get("17業種", "")
                    result["テーマ"] = row.get(
                        "テーマ",
                        infer_theme(code, name, row.get("33業種", ""), row.get("17業種", ""))
                    )
                    results.append(result)

            if scan_weekly:
                weekly_params = WEDGE_WEEKLY_PRESETS[preset_name]
                long_daily_df = download_price_long(code)
                weekly_df = resample_to_weekly_ohlcv(long_daily_df)
                result = analyze_wedge_break(
                    price_df=weekly_df,
                    code=code,
                    name=name,
                    market=market,
                    params=weekly_params,
                    # 200週線は5年分では十分に安定しないため、週足には適用しない
                    use_ma200_filter=False,
                    timeframe_label="週足",
                    ma_unit="週"
                )
                if result is not None:
                    result["33業種"] = row.get("33業種", "")
                    result["17業種"] = row.get("17業種", "")
                    result["テーマ"] = row.get(
                        "テーマ",
                        infer_theme(code, name, row.get("33業種", ""), row.get("17業種", ""))
                    )
                    results.append(result)

        except Exception:
            failed_count += 1

        progress_bar.progress(n / total)
        if wait_sec > 0:
            time.sleep(wait_sec)

    status_area.write("スクリーニング完了")

    result_key = "screen_result_df_wedge"
    failed_key = "screen_failed_count_wedge"
    market_key = "screen_market_wedge"

    if not results:
        st.session_state[result_key] = pd.DataFrame()
    else:
        result_df = pd.DataFrame(results)
        buy_rank = {"候補": 0, "監視": 1, "見送り": 2}
        priority_rank = {"A": 0, "B": 1, "C": 2}
        setup_rank = {"WBPB": 0, "WB": 1, "WB監視": 2}
        timeframe_rank = {"週足": 0, "日足": 1}

        result_df["_買い候補順"] = result_df["買い候補"].map(buy_rank).fillna(9)
        result_df["_優先度順"] = result_df["優先度"].map(priority_rank).fillna(9)
        result_df["_セットアップ順"] = result_df["セットアップ"].map(setup_rank).fillna(9)
        result_df["_足種順"] = result_df["足種"].map(timeframe_rank).fillna(9)

        result_df = result_df.sort_values(
            by=["_買い候補順", "_優先度順", "_足種順", "_セットアップ順", "第2RR", "出来高倍率", "スコア"],
            ascending=[True, True, True, True, False, False, False]
        ).reset_index(drop=True)

        st.session_state[result_key] = result_df

    st.session_state[failed_key] = failed_count
    st.session_state[market_key] = market_choice
    show_saved_wedge_results()


def render_wedge_tab(key_prefix):
    st.subheader("ウェッジブレイク / ウェッジブレイク・プルバック")

    st.write(
        "上昇トレンド後の下降ウェッジを探し、"
        "①下降トレンドラインを上抜けるWB、"
        "②初回ブレイク後の押しからスイング高値を抜くWBPB"
        "を日足・週足で分けて表示します。"
    )

    st.info(
        "週足は長い上昇トレンドの中期セットアップ用です。"
        "週足を選ぶと5年分の日足から週足を作るため、日足より実行時間が長くなります。"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        timeframe_choice = st.selectbox(
            "確認する足種",
            ["日足", "週足", "日足・週足の両方"],
            index=0,
            key=f"{key_prefix}_timeframe"
        )

    with col2:
        preset_name = st.selectbox(
            "判定の厳しさ",
            ["普通", "厳しめ", "ゆるめ"],
            index=1,
            key=f"{key_prefix}_preset"
        )

    with col3:
        params = WEDGE_WEEKLY_PRESETS[preset_name] if timeframe_choice == "週足" else WEDGE_PRESETS[preset_name]
        unit = "週" if timeframe_choice == "週足" else "営業日"
        st.write("現在の主な条件")
        st.write(f"ウェッジ期間：直近{params['lookback']}{unit}")
        st.write(f"前段の上昇：{params['min_prior_rise_pct']}%以上")
        st.write(f"ブレイク出来高：平均の{params['breakout_volume_min']}倍以上目安")
        st.write(f"損切り許容：{params['stop_loss_max_pct']}%以内")

    if timeframe_choice == "日足・週足の両方":
        st.caption("両方を選ぶと、日足と週足それぞれの条件で抽出します。結果の「足種」列で分けて確認できます。")

    run_button = st.button(
        "ウェッジブレイクをスクリーニング実行",
        type="primary",
        key=f"{key_prefix}_run"
    )

    if run_button:
        run_wedge_screener(
            market_choice=market_choice,
            preset_name=preset_name,
            timeframe_choice=timeframe_choice,
            max_scan=max_scan,
            wait_sec=wait_sec,
            use_ma200_filter=use_ma200_filter
        )
    else:
        show_saved_wedge_results()

st.title("25日線・75日線・200日線・ウェッジブレイク スクリーニングアプリ")

st.write("出来高急増後の25日線初押し、75日線押し目、200日線押し目、下降ウェッジのWB/WBPBを探すスクリーニングアプリです。ウェッジは日足・週足に対応しています。")
st.info("日付は固定していません。取得できた株価データの最新取引日で自動判定します。")

with st.sidebar:
    st.header("共通設定")

    market_choice = st.selectbox(
        "市場を選択",
        ["全市場", "プライム", "スタンダード", "グロース"]
    )

    use_volume_filter = st.checkbox("75日線・200日線：出来高が暴れすぎていない銘柄に絞る", value=True)
    use_ma200_filter = st.checkbox("25日線・75日線・日足ウェッジ：200日線より上の銘柄に絞る", value=False)

    st.markdown("### 実行設定")

    max_scan = st.number_input(
        "確認する銘柄数。0なら選択市場すべて",
        min_value=0,
        max_value=5000,
        value=100,
        step=50
    )

    wait_sec = st.number_input(
        "1銘柄ごとの待機秒数",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.1
    )

    if st.button("キャッシュをクリア"):
        st.cache_data.clear()
        st.success("キャッシュをクリアしました。")


with st.expander("このアプリの見方"):
    st.write(
        """
        25日線初押しは、前日比5倍以上の出来高急増を起点に、数日上昇した後、初めて25日線付近まで押してきた銘柄を探します。
        75日線押し目は、中期上昇トレンドにおけるより深い押し目を探します。
        200日線押し目は、長期上昇トレンドの中で200日線付近まで大きく押した銘柄を探します。
        ウェッジブレイクは、上昇トレンド後の下降ウェッジを対象に、下降トレンドライン上抜けのWBと、初回ブレイク後の押しから再度スイング高値を抜くWBPBを探します。

        ウェッジの日足は短期〜数週間のスイング候補、週足は数週間〜数か月の中期候補を探す用途です。
        週足は長期データを使うため、最初は確認銘柄数を100程度にして動作確認してください。

        「全市場」はプライム・スタンダード・グロースをまとめて確認します。
        ETF、REIT、TOKYO PRO Marketなどは対象外です。

        どのスクリーニングも抽出後にチャートで、支持線・抵抗線、出来高、地合い、材料を確認してから売買判断してください。
        """
    )

tab25, tab75, tab200, tab_wedge = st.tabs(["25日線初押し", "75日線押し目", "200日線押し目", "ウェッジブレイク（日足・週足）"])

with tab25:
    render_tab(25, "ma25")

with tab75:
    render_tab(75, "ma75")

with tab200:
    render_tab(200, "ma200")

with tab_wedge:
    render_wedge_tab("wedge")
