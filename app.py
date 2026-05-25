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

    return {
        "コード": code,
        "銘柄名": name,
        "市場": market,
        "判定": f"{ma_period}日線押し目",
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

    st.subheader("抽出結果")

    if len(results) == 0:
        st.warning("条件に一致する銘柄は見つかりませんでした。条件を「ゆるめ」にするか、確認銘柄数を増やしてください。")
    else:
        result_df = pd.DataFrame(results)

        sort_cols = ["スコア", f"{ma_period}日線傾き%", f"{ma_period}日線距離%"]
        result_df = result_df.sort_values(
            by=sort_cols,
            ascending=[False, False, True]
        ).reset_index(drop=True)

        st.write(f"抽出銘柄数：{len(result_df)}")
        st.write(f"取得失敗数：{failed_count}")

        st.dataframe(result_df, use_container_width=True, hide_index=True)

        csv = result_df.to_csv(index=False, encoding="utf-8-sig")

        market_for_filename = market_choice.replace("全市場", "all_markets")

        st.download_button(
            label="CSVダウンロード",
            data=csv,
            file_name=f"ma{ma_period}_pullback_{market_for_filename}.csv",
            mime="text/csv",
            key=f"download_ma{ma_period}_{market_for_filename}"
        )


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


st.title("移動平均線押し目スクリーニングアプリ")

st.write("25日線押し目と75日線押し目を、タブで切り替えて探せます。")
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

        「全市場」は、プライム・スタンダード・グロースをまとめて確認します。
        ETF、REIT、TOKYO PRO Marketなどは対象外にしています。

        最初は確認銘柄数を100にして動作確認し、問題なければ0にして市場全体を確認してください。
        """
    )

tab25, tab75 = st.tabs(["25日線押し目", "75日線押し目"])

with tab25:
    render_tab(25, "ma25")

with tab75:
    render_tab(75, "ma75")
