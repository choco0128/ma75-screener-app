import time
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


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
    page_title="25日線初押し・75日線押し目・三角保ち合い・カップウィズハンドル",
    layout="wide"
)


def find_column(columns, keyword):
    for col in columns:
        if keyword in str(col):
            return col
    return None


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

    result = df[[code_col, name_col, market_col]].copy()
    result.columns = ["コード", "銘柄名", "市場"]

    result["コード"] = (
        result["コード"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.upper()
    )

    result["銘柄名"] = result["銘柄名"].astype(str).str.strip()
    result["市場"] = result["市場"].astype(str).str.strip()

    result = result[result["コード"].str.match(r"^[0-9A-Z]{4}$", na=False)]
    return result.drop_duplicates(subset=["コード"]).reset_index(drop=True)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_price(code):
    ticker = f"{code}.T"

    df = yf.download(
        ticker,
        period="1y",
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
    else:
        recent_low = low.tail(10).min()
        ma_stop = latest_ma75 * 0.98 if pd.notna(latest_ma75) else np.nan

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
    else:
        st.subheader("75日線押し目スクリーニング")
        st.write("中期上昇トレンド中に、75日線まで深めに押している銘柄を探します。")

    col1, col2 = st.columns(2)

    with col1:
        preset_name = st.selectbox(
            "判定の厳しさ",
            ["普通", "厳しめ", "ゆるめ"],
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
            ["普通", "厳しめ", "ゆるめ"],
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
            ["普通", "厳しめ", "ゆるめ"],
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

st.title("25日線初押し・75日線押し目・三角保ち合い・カップウィズハンドル スクリーニングアプリ")

st.write("出来高急増後の25日線初押し・75日線押し目・三角保ち合いブレイク初動・カップウィズハンドル上抜け接近をタブで切り替えて探せます。")
st.info("日付は固定していません。取得できた株価データの最新取引日で自動判定します。")

with st.sidebar:
    st.header("共通設定")

    market_choice = st.selectbox(
        "市場を選択",
        ["全市場", "プライム", "スタンダード", "グロース"]
    )

    use_volume_filter = st.checkbox("75日線：出来高が暴れすぎていない銘柄に絞る", value=True)
    use_ma200_filter = st.checkbox("200日線より上の銘柄に絞る", value=False)

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
        25日線初押しは、前日比5倍以上の出来高急増を起点に、数日上昇した後、初めて25日線付近まで押してきた銘柄を探す用途です。
        75日線押し目は、従来通り、より深い押し目や中期トレンドの押し目を探す用途です。
        三角保ち合いブレイクは、上昇トレンド中に値幅と出来高が縮小し、出来高増で上抜ける初動を探す用途です。
        カップウィズハンドルは、日足・週足・月足それぞれで、カップ形成後に浅いハンドルを作り、ピボット上抜けが近い銘柄を探す用途です。

        「全市場」は、プライム・スタンダード・グロースをまとめて確認します。
        ETF、REIT、TOKYO PRO Marketなどは対象外にしています。

        25日線初押しの実践ルール：
        ・起点：直近で出来高が前日比5倍以上
        ・上昇：起点後に数日で一定以上上昇
        ・調整：高値から下落して25日線付近へ接近
        ・初押し：急騰後、今回が最初の25日線接近に近いもの
        ・買値：直近5日高値を上抜け
        ・損切り：25日線の少し下、または直近安値割れ
        ・第1利確：起点後高値
        ・第2利確：起点後高値から上振れ分

        最初は確認銘柄数を100にして動作確認し、問題なければ0にして市場全体を確認してください。
        """
    )

tab25, tab75, tab_triangle, tab_cup = st.tabs(["25日線初押し", "75日線押し目", "三角保ち合いブレイク", "カップウィズハンドル"])

with tab25:
    render_tab(25, "ma25")

with tab75:
    render_tab(75, "ma75")

with tab_triangle:
    render_triangle_tab("triangle")

with tab_cup:
    render_cup_tab("cup")
