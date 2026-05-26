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
            "near_lower": -1.5,
            "near_upper": 2.5,
            "slope_lookback": 10,
            "slope_min": 0.8,
            "along_days": 4,
            "min_near_days": 2,
            "volume_ratio_max": 1.6,
        },
        "普通": {
            "near_lower": -2.5,
            "near_upper": 4.0,
            "slope_lookback": 10,
            "slope_min": 0.3,
            "along_days": 4,
            "min_near_days": 2,
            "volume_ratio_max": 2.0,
        },
        "ゆるめ": {
            "near_lower": -4.0,
            "near_upper": 6.0,
            "slope_lookback": 10,
            "slope_min": 0.0,
            "along_days": 4,
            "min_near_days": 1,
            "volume_ratio_max": 2.8,
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


TRIANGLE_PRESETS = {
    "厳しめ": {
        "lookback": 40,
        "high_slope_max_pct_per_day": 0.00,
        "low_slope_min_pct_per_day": 0.02,
        "range_ratio_max": 0.75,
        "volume_contract_ratio_max": 0.85,
        "breakout_volume_min": 2.0,
        "close_position_min": 0.55,
        "stop_loss_max_pct": 8.0,
        "rr_min": 1.2,
        "need_ma75_up": True,
    },
    "普通": {
        "lookback": 40,
        "high_slope_max_pct_per_day": 0.08,
        "low_slope_min_pct_per_day": -0.02,
        "range_ratio_max": 0.85,
        "volume_contract_ratio_max": 1.00,
        "breakout_volume_min": 1.5,
        "close_position_min": 0.50,
        "stop_loss_max_pct": 10.0,
        "rr_min": 1.0,
        "need_ma75_up": False,
    },
    "ゆるめ": {
        "lookback": 30,
        "high_slope_max_pct_per_day": 0.18,
        "low_slope_min_pct_per_day": -0.08,
        "range_ratio_max": 0.95,
        "volume_contract_ratio_max": 1.20,
        "breakout_volume_min": 1.2,
        "close_position_min": 0.45,
        "stop_loss_max_pct": 12.0,
        "rr_min": 0.8,
        "need_ma75_up": False,
    },
}


st.set_page_config(
    page_title="移動平均線押し目スクリーニング",
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
    st.subheader(f"{ma_period}日線押し目スクリーニング")

    if ma_period == 25:
        st.write("短期〜中期の上昇トレンド中に、25日線まで押している銘柄を探します。")
    else:
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
        st.write(f"{ma_period}日線との距離：{params['near_lower']}% 〜 +{params['near_upper']}%")
        st.write(f"{ma_period}日線の傾き：{params['slope_min']}%以上")
        st.write(f"直近{params['along_days']}日中、{params['min_near_days']}日以上が{ma_period}日線付近")

    run_button = st.button(
        f"{ma_period}日線スクリーニング実行",
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


st.title("押し目・三角保ち合いブレイク スクリーニングアプリ")

st.write("25日線押し目・75日線押し目・三角保ち合いブレイク初動を、タブで切り替えて探せます。")
st.info("日付は固定していません。取得できた株価データの最新取引日で自動判定します。")

with st.sidebar:
    st.header("共通設定")

    market_choice = st.selectbox(
        "市場を選択",
        ["全市場", "プライム", "スタンダード", "グロース"]
    )

    use_volume_filter = st.checkbox("出来高が暴れすぎていない銘柄に絞る", value=True)
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
        25日線押し目は、比較的短期の上昇トレンドの押し目を探す用途です。
        75日線押し目は、より深い押し目や中期トレンドの押し目を探す用途です。
        三角保ち合いブレイクは、上昇トレンド中に値幅と出来高が縮小し、出来高増で上抜ける初動を探す用途です。

        「全市場」は、プライム・スタンダード・グロースをまとめて確認します。
        ETF、REIT、TOKYO PRO Marketなどは対象外にしています。

        実践ルール：
        ・買値：直近5日高値を上抜け
        ・損切り：25日線/75日線の少し下、または直近安値割れ
        ・第1利確：直近20日高値
        ・第2利確：直近60日高値
        ・買い候補：第1RR 1.5以上、損切り5%以内、反発確認あり
        ・監視：第1RR 1.0以上、または第2RR 1.5以上
        ・見送り：RR悪い、反発なし、損切り遠い、危険サイン複数

        最初は確認銘柄数を100にして動作確認し、問題なければ0にして市場全体を確認してください。
        """
    )

tab25, tab75, tab_triangle = st.tabs(["25日線押し目", "75日線押し目", "三角保ち合いブレイク"])

with tab25:
    render_tab(25, "ma25")

with tab75:
    render_tab(75, "ma75")

with tab_triangle:
    render_triangle_tab("triangle")
