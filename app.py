import re
import time
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# =========================
# 基本設定
# =========================
st.set_page_config(
    page_title="三角保ち合い・移動平均ブレイク判定アプリ",
    layout="wide"
)

JPX_EXCEL_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"


# =========================
# 共通関数
# =========================
def normalize_code(code):
    """
    日本株コードを yfinance 用の形式に変換する。
    例:
    4506 -> 4506.T
    7564 -> 7564.T
    141A -> 141A.T
    """
    if pd.isna(code):
        return None

    code = str(code).strip().upper()
    code = code.replace(".T", "")
    code = re.sub(r"[^0-9A-Z]", "", code)

    if not code:
        return None

    return f"{code}.T"


def display_code(symbol):
    """
    4506.T -> 4506
    """
    return str(symbol).replace(".T", "")


@st.cache_data(ttl=60 * 60)
def fetch_stock_data(symbol, period="1y"):
    """
    yfinanceから日足データを取得する。
    """
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return None

        # yfinanceの列がMultiIndexになる場合の対策
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in needed):
            return None

        df = df[needed].copy()
        df = df.dropna()

        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()

        if len(df) < 90:
            return None

        return df

    except Exception:
        return None


@st.cache_data(ttl=60 * 60 * 24)
def load_jpx_list():
    """
    JPXの上場銘柄一覧を取得する。
    Streamlit Cloud上で動かす場合、requirements.txtにxlrdが必要。
    """
    df = pd.read_excel(JPX_EXCEL_URL)

    # 想定列：
    # コード / 銘柄名 / 市場・商品区分
    required_cols = ["コード", "銘柄名", "市場・商品区分"]
    if not all(c in df.columns for c in required_cols):
        return pd.DataFrame(columns=["コード", "銘柄名", "市場"])

    out = df[required_cols].copy()
    out.columns = ["コード", "銘柄名", "市場"]

    # ETF等を除きやすくするため、普通株に近いものを残す
    out = out[out["市場"].astype(str).str.contains("プライム|スタンダード|グロース", na=False)]

    out["コード"] = out["コード"].astype(str)
    out["symbol"] = out["コード"].apply(normalize_code)
    out = out.dropna(subset=["symbol"])
    out = out.drop_duplicates(subset=["symbol"])

    return out


def calc_volume_ratio(df, days=5):
    """
    直近出来高 ÷ 直近日の前までの平均出来高
    """
    if len(df) < days + 1:
        return 0

    last_volume = df["Volume"].iloc[-1]
    avg_volume = df["Volume"].iloc[-days - 1:-1].mean()

    if avg_volume <= 0:
        return 0

    return last_volume / avg_volume


def calc_close_position(last):
    """
    当日レンジ内で終値がどの位置か。
    1.0に近いほど高値引け。
    """
    high = last["High"]
    low = last["Low"]

    if high == low:
        return 0.5

    return (last["Close"] - low) / (high - low)


# =========================
# 25日線・75日線ブレイク判定
# =========================
def detect_ma_breakout(df, ma=25, volume_mult=1.5):
    if df is None or len(df) < ma + 10:
        return None

    df = df.copy()
    df[f"MA{ma}"] = df["Close"].rolling(ma).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    current_ma = last[f"MA{ma}"]
    prev_ma = prev[f"MA{ma}"]

    if pd.isna(current_ma) or pd.isna(prev_ma):
        return None

    ma_up = current_ma > df[f"MA{ma}"].iloc[-6]
    close_above_ma = last["Close"] > current_ma
    crossed_today = prev["Close"] <= prev_ma and last["Close"] > current_ma
    volume_ratio = calc_volume_ratio(df, days=5)
    is_bullish = last["Close"] > last["Open"]
    close_position = calc_close_position(last)

    stop_loss = current_ma
    risk = last["Close"] - stop_loss

    # 仮の利確目安：リスクの2倍
    target_price = last["Close"] + risk * 2 if risk > 0 else np.nan
    rr = 2.0 if risk > 0 else 0

    reasons = []
    if not close_above_ma:
        reasons.append(f"終値が{ma}日線を上回っていない")
    if not ma_up:
        reasons.append(f"{ma}日線が上向きではない")
    if volume_ratio < volume_mult:
        reasons.append("出来高倍率が不足")
    if not is_bullish:
        reasons.append("陰線")
    if close_position < 0.5:
        reasons.append("終値が当日レンジの上半分ではない")

    if crossed_today and ma_up and volume_ratio >= volume_mult and is_bullish:
        decision = "買い候補"
        reason = f"{ma}日線を出来高増で上抜け"
    elif close_above_ma and ma_up:
        decision = "監視候補"
        reason = f"{ma}日線より上で推移"
    else:
        decision = "見送り"
        reason = " / ".join(reasons) if reasons else "条件不足"

    return {
        "判定": decision,
        "理由": reason,
        "現在値": round(float(last["Close"]), 1),
        "基準ライン": round(float(current_ma), 1),
        "出来高倍率": round(float(volume_ratio), 2),
        "損切りライン": round(float(stop_loss), 1),
        "利確目安": round(float(target_price), 1) if pd.notna(target_price) else None,
        "リスクリワード": round(float(rr), 2),
        "陽線": bool(is_bullish),
        "終値位置": round(float(close_position), 2),
        "MA上向き": bool(ma_up),
    }


# =========================
# 三角保ち合いブレイク判定
# =========================
def detect_triangle_breakout(
    df,
    lookback=40,
    volume_mult=2.0,
    breakout_margin_pct=0.0,
    max_stop_loss_pct=8.0,
):
    """
    三角保ち合いからの出来高増ブレイクを判定する。
    """
    if df is None or len(df) < max(lookback + 10, 90):
        return None

    df = df.copy()

    df["MA25"] = df["Close"].rolling(25).mean()
    df["MA75"] = df["Close"].rolling(75).mean()

    recent = df.tail(lookback)
    last = df.iloc[-1]

    if recent.empty or pd.isna(last["MA25"]) or pd.isna(last["MA75"]):
        return None

    x = np.arange(len(recent))

    # 高値ライン・安値ラインを一次回帰で近似
    high_slope, high_intercept = np.polyfit(x, recent["High"], 1)
    low_slope, low_intercept = np.polyfit(x, recent["Low"], 1)

    upper_line = high_slope * (lookback - 1) + high_intercept
    lower_line = low_slope * (lookback - 1) + low_intercept

    price_base = recent["Close"].mean()
    high_slope_pct = high_slope / price_base if price_base > 0 else 0
    low_slope_pct = low_slope / price_base if price_base > 0 else 0

    first_half = recent.iloc[:lookback // 2]
    second_half = recent.iloc[lookback // 2:]

    first_range = first_half["High"].max() - first_half["Low"].min()
    second_range = second_half["High"].max() - second_half["Low"].min()
    range_shrinking = second_range < first_range * 0.80 if first_range > 0 else False

    first_volume = first_half["Volume"].mean()
    second_volume = second_half["Volume"].mean()
    volume_shrinking = second_volume < first_volume if first_volume > 0 else False

    volume_ratio = calc_volume_ratio(df, days=5)

    ma25_up = df["MA25"].iloc[-1] > df["MA25"].iloc[-6]
    ma75_up = df["MA75"].iloc[-1] > df["MA75"].iloc[-6]
    above_ma25 = last["Close"] > df["MA25"].iloc[-1]
    above_ma75 = last["Close"] > df["MA75"].iloc[-1]

    is_bullish = last["Close"] > last["Open"]
    close_position = calc_close_position(last)

    # 三角保ち合い条件
    # 高値ラインは横ばい〜やや切り下げを許容
    # 安値ラインは横ばい〜切り上げを許容
    is_triangle = (
        high_slope_pct <= 0.0015 and
        low_slope_pct >= -0.0005 and
        range_shrinking and
        volume_shrinking and
        above_ma75
    )

    breakout_line = upper_line * (1 + breakout_margin_pct / 100)
    is_breakout = (
        last["Close"] > breakout_line and
        volume_ratio >= volume_mult and
        is_bullish and
        close_position >= 0.50
    )

    # 損切りは下限ライン、ただし遠すぎる場合は直近10日安値も参考
    recent_10_low = df["Low"].tail(10).min()
    stop_loss = max(lower_line, recent_10_low)

    risk = last["Close"] - stop_loss
    stop_loss_pct = risk / last["Close"] * 100 if last["Close"] > 0 else 999

    triangle_width = max(upper_line - lower_line, 0)
    target_price = last["Close"] + triangle_width
    reward = target_price - last["Close"]
    rr = reward / risk if risk > 0 else 0

    reasons = []

    if not is_triangle:
        if high_slope_pct > 0.0015:
            reasons.append("高値ラインが上がりすぎて三角保ち合いになりにくい")
        if low_slope_pct < -0.0005:
            reasons.append("安値が切り上がっていない")
        if not range_shrinking:
            reasons.append("値幅縮小が弱い")
        if not volume_shrinking:
            reasons.append("保ち合い中の出来高減少が弱い")
        if not above_ma75:
            reasons.append("株価が75日線より下")

    if is_triangle and not is_breakout:
        if last["Close"] <= breakout_line:
            reasons.append("まだ上限ラインを終値で上抜けていない")
        if volume_ratio < volume_mult:
            reasons.append("ブレイク出来高が不足")
        if not is_bullish:
            reasons.append("陽線ではない")
        if close_position < 0.50:
            reasons.append("終値位置が弱い")

    if stop_loss_pct > max_stop_loss_pct:
        reasons.append("損切りラインまでが遠い")

    if is_triangle and is_breakout and stop_loss_pct <= max_stop_loss_pct:
        decision = "買い候補"
        reason = "三角保ち合いを出来高増で上抜け"
    elif is_triangle and not is_breakout:
        decision = "ブレイク前候補"
        reason = "三角保ち合い中だが、まだ上抜け前"
    elif is_triangle and is_breakout and stop_loss_pct > max_stop_loss_pct:
        decision = "見送り"
        reason = "上抜けはしているが損切りまでが遠い"
    else:
        decision = "見送り"
        reason = " / ".join(reasons) if reasons else "三角保ち合い条件を満たさない"

    return {
        "判定": decision,
        "理由": reason,
        "現在値": round(float(last["Close"]), 1),
        "上抜けライン": round(float(breakout_line), 1),
        "保ち合い上限": round(float(upper_line), 1),
        "保ち合い下限": round(float(lower_line), 1),
        "出来高倍率": round(float(volume_ratio), 2),
        "損切りライン": round(float(stop_loss), 1),
        "損切り率%": round(float(stop_loss_pct), 2),
        "利確目安": round(float(target_price), 1),
        "リスクリワード": round(float(rr), 2),
        "25日線上向き": bool(ma25_up),
        "75日線上向き": bool(ma75_up),
        "株価75日線上": bool(above_ma75),
        "値幅縮小": bool(range_shrinking),
        "出来高減少": bool(volume_shrinking),
        "陽線": bool(is_bullish),
        "終値位置": round(float(close_position), 2),
    }


# =========================
# 銘柄リスト作成
# =========================
def build_universe_from_text(raw_text):
    codes = []
    for part in re.split(r"[\s,、\n]+", raw_text):
        part = part.strip()
        if not part:
            continue
        symbol = normalize_code(part)
        if symbol:
            codes.append({
                "コード": display_code(symbol),
                "銘柄名": "",
                "市場": "",
                "symbol": symbol,
            })

    df = pd.DataFrame(codes)
    if df.empty:
        return df

    return df.drop_duplicates(subset=["symbol"])


def build_universe_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)

    code_col = None
    for c in df.columns:
        if str(c).lower() in ["コード", "code", "ticker", "銘柄コード"]:
            code_col = c
            break

    if code_col is None:
        st.error("CSVに「コード」列、または code 列が必要です。")
        return pd.DataFrame(columns=["コード", "銘柄名", "市場", "symbol"])

    name_col = None
    for c in df.columns:
        if str(c).lower() in ["銘柄名", "name"]:
            name_col = c
            break

    market_col = None
    for c in df.columns:
        if str(c).lower() in ["市場", "market"]:
            market_col = c
            break

    out = pd.DataFrame()
    out["コード"] = df[code_col].astype(str)
    out["銘柄名"] = df[name_col].astype(str) if name_col else ""
    out["市場"] = df[market_col].astype(str) if market_col else ""
    out["symbol"] = out["コード"].apply(normalize_code)
    out = out.dropna(subset=["symbol"])
    out = out.drop_duplicates(subset=["symbol"])

    return out


# =========================
# Session State
# =========================
if "results" not in st.session_state:
    st.session_state.results = None

if "selected_code" not in st.session_state:
    st.session_state.selected_code = None


# =========================
# 画面
# =========================
st.title("三角保ち合い・移動平均ブレイク判定アプリ")

st.caption(
    "25日線・75日線ブレイクに加えて、トライアル・住友ファーマ・ワークマンのような"
    "三角保ち合いからの出来高増ブレイク初動を抽出するためのアプリです。"
)

with st.sidebar:
    st.header("スクリーニング設定")

    mode = st.selectbox(
        "スクリーニング種類",
        [
            "25日線ブレイク",
            "75日線ブレイク",
            "三角保ち合いブレイク初動",
        ],
    )

    period = st.selectbox(
        "取得期間",
        ["6mo", "1y", "2y"],
        index=1,
    )

    st.divider()

    if mode in ["25日線ブレイク", "75日線ブレイク"]:
        ma_volume_mult = st.selectbox(
            "出来高倍率",
            [1.0, 1.5, 2.0, 3.0],
            index=1,
        )
    else:
        lookback = st.selectbox(
            "三角保ち合いを見る期間",
            [20, 30, 40, 60],
            index=2,
        )

        triangle_volume_mult = st.selectbox(
            "ブレイク時の出来高倍率",
            [1.5, 2.0, 3.0],
            index=1,
        )

        breakout_margin_pct = st.selectbox(
            "上抜け余裕幅",
            [0.0, 0.5, 1.0],
            index=0,
            help="0.5なら、上限ラインより0.5%以上上で終値を付けた場合に上抜けと判定します。",
        )

        max_stop_loss_pct = st.selectbox(
            "損切り許容幅",
            [5.0, 8.0, 10.0, 12.0],
            index=1,
        )

    st.divider()

    display_target = st.selectbox(
        "表示対象",
        [
            "買い候補のみ",
            "買い候補＋監視候補",
            "全部表示",
        ],
        index=1,
    )


st.subheader("対象銘柄")

source_type = st.radio(
    "銘柄の指定方法",
    [
        "コードを直接入力",
        "CSVアップロード",
        "JPX一覧から取得",
    ],
    horizontal=True,
)

universe_df = pd.DataFrame(columns=["コード", "銘柄名", "市場", "symbol"])

if source_type == "コードを直接入力":
    default_codes = "4506, 7564, 141A"
    raw_codes = st.text_area(
        "銘柄コードを入力",
        value=default_codes,
        height=100,
        help="例：4506, 7564, 141A のように入力。改行区切りでもOKです。",
    )
    universe_df = build_universe_from_text(raw_codes)

elif source_type == "CSVアップロード":
    uploaded = st.file_uploader(
        "CSVをアップロード",
        type=["csv"],
        help="「コード」列が必要です。任意で「銘柄名」「市場」列も使えます。",
    )
    if uploaded is not None:
        universe_df = build_universe_from_csv(uploaded)

else:
    st.info("JPX一覧取得は便利ですが、全銘柄を一度に見ると時間がかかります。最初は市場と件数を絞るのがおすすめです。")
    try:
        jpx_df = load_jpx_list()

        market_options = ["すべて", "プライム", "スタンダード", "グロース"]
        market_choice = st.selectbox("市場", market_options, index=3)

        if market_choice != "すべて":
            jpx_df = jpx_df[jpx_df["市場"].astype(str).str.contains(market_choice, na=False)]

        max_count = st.slider(
            "処理する最大銘柄数",
            min_value=20,
            max_value=1000,
            value=200,
            step=20,
            help="Streamlit Cloudでは処理数が多いとタイムアウトすることがあります。",
        )

        universe_df = jpx_df.head(max_count).copy()

        st.write(f"対象銘柄数：{len(universe_df)}")

    except Exception as e:
        st.error(f"JPX一覧の取得に失敗しました：{e}")


if not universe_df.empty:
    st.write("読み込み銘柄")
    st.dataframe(
        universe_df[["コード", "銘柄名", "市場"]].head(50),
        use_container_width=True,
        hide_index=True,
    )


# =========================
# スクリーニング実行
# =========================
run = st.button("スクリーニング実行", type="primary")

if run:
    if universe_df.empty:
        st.error("対象銘柄がありません。")
    else:
        results = []
        progress = st.progress(0)
        status = st.empty()

        total = len(universe_df)

        for i, row in universe_df.reset_index(drop=True).iterrows():
            symbol = row["symbol"]
            code = row["コード"]
            name = row.get("銘柄名", "")
            market = row.get("市場", "")

            status.write(f"処理中：{code} {name}  ({i + 1}/{total})")

            df = fetch_stock_data(symbol, period=period)

            if df is not None:
                if mode == "25日線ブレイク":
                    result = detect_ma_breakout(
                        df,
                        ma=25,
                        volume_mult=ma_volume_mult,
                    )
                elif mode == "75日線ブレイク":
                    result = detect_ma_breakout(
                        df,
                        ma=75,
                        volume_mult=ma_volume_mult,
                    )
                else:
                    result = detect_triangle_breakout(
                        df,
                        lookback=lookback,
                        volume_mult=triangle_volume_mult,
                        breakout_margin_pct=breakout_margin_pct,
                        max_stop_loss_pct=max_stop_loss_pct,
                    )

                if result is not None:
                    result["コード"] = code
                    result["銘柄名"] = name
                    result["市場"] = market
                    result["symbol"] = symbol
                    results.append(result)

            progress.progress((i + 1) / total)

        status.empty()

        if not results:
            st.warning("判定できる銘柄がありませんでした。")
            st.session_state.results = None
        else:
            result_df = pd.DataFrame(results)

            # 表示対象で絞り込み
            if display_target == "買い候補のみ":
                result_df = result_df[result_df["判定"] == "買い候補"]
            elif display_target == "買い候補＋監視候補":
                result_df = result_df[
                    result_df["判定"].isin(["買い候補", "監視候補", "ブレイク前候補"])
                ]

            # 並び替え
            sort_cols = []
            if "リスクリワード" in result_df.columns:
                sort_cols.append("リスクリワード")
            if "出来高倍率" in result_df.columns:
                sort_cols.append("出来高倍率")

            if sort_cols and not result_df.empty:
                result_df = result_df.sort_values(sort_cols, ascending=False)

            st.session_state.results = result_df.reset_index(drop=True)


# =========================
# 結果表示
# =========================
if st.session_state.results is not None:
    result_df = st.session_state.results.copy()

    st.subheader("抽出結果")

    if result_df.empty:
        st.warning("条件に合う銘柄はありませんでした。条件を少し緩めて再実行してください。")
    else:
        main_cols = [
            c for c in [
                "コード",
                "銘柄名",
                "市場",
                "判定",
                "理由",
                "現在値",
                "上抜けライン",
                "基準ライン",
                "出来高倍率",
                "損切りライン",
                "損切り率%",
                "利確目安",
                "リスクリワード",
            ] if c in result_df.columns
        ]

        st.dataframe(
            result_df[main_cols],
            use_container_width=True,
            hide_index=True,
        )

        csv = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "結果をCSVでダウンロード",
            data=csv,
            file_name="screening_results.csv",
            mime="text/csv",
        )

        st.subheader("銘柄詳細")

        options = result_df["コード"].astype(str).tolist()
        selected = st.selectbox(
            "詳細を見る銘柄",
            options,
            index=0,
            key="selected_detail_code",
        )

        st.session_state.selected_code = selected

        detail = result_df[result_df["コード"].astype(str) == str(selected)].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("判定", detail.get("判定", "-"))
        col2.metric("現在値", detail.get("現在値", "-"))
        col3.metric("出来高倍率", detail.get("出来高倍率", "-"))
        col4.metric("リスクリワード", detail.get("リスクリワード", "-"))

        st.write("### 判断理由")
        st.write(detail.get("理由", ""))

        line_cols = [
            c for c in [
                "上抜けライン",
                "保ち合い上限",
                "保ち合い下限",
                "基準ライン",
                "損切りライン",
                "損切り率%",
                "利確目安",
                "25日線上向き",
                "75日線上向き",
                "株価75日線上",
                "値幅縮小",
                "出来高減少",
                "陽線",
                "終値位置",
            ] if c in result_df.columns
        ]

        st.write("### 詳細データ")
        st.dataframe(
            pd.DataFrame(detail[line_cols]).rename(columns={detail.name: "値"}),
            use_container_width=True,
        )

        # 簡易チャート
        symbol = detail.get("symbol")
        if symbol:
            chart_df = fetch_stock_data(symbol, period=period)
            if chart_df is not None:
                chart_data = chart_df[["Close"]].copy()
                chart_data["MA25"] = chart_df["Close"].rolling(25).mean()
                chart_data["MA75"] = chart_df["Close"].rolling(75).mean()
                st.write("### 簡易チャート")
                st.line_chart(chart_data)

st.caption("注意：このアプリは投資判断を補助するスクリーニングツールです。売買判断は必ずチャート・材料・地合いと合わせて確認してください。")
