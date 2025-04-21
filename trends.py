import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pytrends.request import TrendReq
from prophet import Prophet
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import streamlit as st
from datetime import datetime
import time
import requests
from urllib.parse import quote
import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_user, smtp_password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(from_email, [to_email], msg.as_string())

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trends_cached(keywords, geo=None, timeframe='today 1-d'):
    return fetch_trends(keywords, geo, timeframe)

def fetch_trends(keywords, geo=None, timeframe='today 1-d'):
    retries = 5
    delay = 5
    for attempt in range(retries):
        try:
            keywords = [quote(str(kw).strip()) for kw in ([keywords] if isinstance(keywords, str) else keywords)]
            pytrends = TrendReq(
                hl='en-US', tz=360,
                requests_args={'headers': {'User-Agent': 'Mozilla/5.0'}})
            pytrends.build_payload(kw_list=keywords, cat=0, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if df.empty:
                st.warning("⚠️ No data available for this query.")
                return pd.DataFrame()
            return df.drop(columns=['isPartial']) if 'isPartial' in df.columns else df
        except Exception as e:
            if "429" in str(e):
                if attempt < retries - 1:
                    wait_time = delay * (2 ** attempt)
                    st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error(f"Rate limit exceeded after {retries} attempts. Please try again later.")
                    return pd.DataFrame()
            else:
                st.error(f"🚨 API Error: {str(e)}")
                return pd.DataFrame()

def forecast_keyword(df, keyword):
    if df.empty or keyword not in df.columns:
        st.warning("⚠️ Not enough data to forecast")
        return None, None
    prophet_df = df.reset_index()[['date', keyword]].rename(columns={'date': 'ds', keyword: 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return model, forecast

def detect_anomalies(df, keyword, sensitivity='Normal'):
    if keyword not in df.columns:
        raise ValueError(f"Keyword '{keyword}' not found in DataFrame.")

    data = df[[keyword]].dropna().copy()
    data['smoothed'] = data[keyword].rolling(window=3, center=True).mean()
    data['smoothed'].fillna(method='bfill', inplace=True)
    data['smoothed'].fillna(method='ffill', inplace=True)

    try:
        stl = STL(data['smoothed'], period=24, robust=True)
        res = stl.fit()
        features = pd.DataFrame({
            'residual': res.resid,
            'trend': res.trend,
            'seasonal': res.seasonal
        }, index=data.index)
    except Exception as e:
        if 'st' in globals():
            st.warning(f"STL decomposition failed: {str(e)}")
        features = pd.DataFrame({'residual': data['smoothed']}, index=data.index)

    features['rolling_mean'] = data['smoothed'].rolling(window=5).mean()
    features['rolling_std'] = data['smoothed'].rolling(window=5).std()
    features.fillna(method='bfill', inplace=True)
    features.fillna(method='ffill', inplace=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    nu_levels = {
        'Strict': 0.01,
        'Normal': 0.05,
        'Sensitive': 0.10
    }
    nu_value = nu_levels.get(sensitivity, 0.05)

    model = OneClassSVM(nu=nu_value, kernel='rbf', gamma='auto')
    preds = model.fit_predict(features_scaled)
    data['anomaly'] = np.where(preds == -1, 1, 0)

    return data[['anomaly']]

def generate_insight_cards(df, keyword):
    cards = []
    if not df.empty:
        forecast_model, forecast = forecast_keyword(df, keyword)
        if forecast is not None:
            peak_date = forecast[forecast['yhat'] == forecast['yhat'].max()]['ds'].iloc[0]
            cards.append(f"📊 **Forecast Peak**: {peak_date.strftime('%d %b')}")
        else:
            cards.append("📊 **Forecast Peak**: N/A")
        current = df[keyword].iloc[-1]
        avg = df[keyword].mean()
        cards.append(f"🚀 **Growth vs Average**: {((current/avg)-1)*100:.1f}%")
        anomalies = detect_anomalies(df, keyword)
        has_anomaly = "Yes" if anomalies['anomaly'].any() else "No"
        cards.append(f"🔔 **Anomaly Detected**: {has_anomaly}")
    return cards

def plot_trend_heatmap(df, keyword):
    df_copy = df.copy()
    df_copy['pct_change'] = df_copy[keyword].pct_change() * 100
    df_copy['hour'] = df_copy.index.hour
    df_copy['date_only'] = df_copy.index.date 
    heatmap_data = df_copy.pivot_table(
        values='pct_change', index='hour', columns='date_only', aggfunc='mean'
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='RdYlGn', center=0, annot=True, fmt='.1f')
    plt.title(f"Hourly Trend Changes for '{keyword}'")
    return plt.gcf()

def get_trend_chart(df, keyword):
    import altair as alt
    return alt.Chart(df.reset_index()).mark_line().encode(
        x='date:T', y=f'{keyword}:Q', tooltip=['date', keyword]
    ).properties(title=f"Search Interest Trend for '{keyword}'").interactive()

def plot_anomalies(df, anomalies, keyword):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[keyword], label='Normal', color='blue')
    anomaly_mask = anomalies['anomaly'] == 1
    ax.scatter(df.index[anomaly_mask], df[keyword][anomaly_mask], color='red', label='Anomaly')
    ax.set_title(f"Anomaly Detection: {keyword}")
    ax.legend()
    return fig

def dashboard():
    st.set_page_config(page_title="Trends Analyzer Pro", layout="wide")
    st.title("🔍 Google Trends Intelligence Suite")

    with st.container():
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            keyword = st.text_input("🔎 Primary Keyword Focus:", "Tesla")
        with col2:
            geo = st.selectbox("🌍 Target Region:", ["GLOBAL", "US", "IN", "GB", "DE", "JP"])
        with col3:
            timeframe_options = {
                "1 Hour": "now 1-H",
                "4 Hours": "now 4-H",
                "1 Day": "now 1-d",
                "7 Days": "now 7-d",
                "30 Days": "today 1-m",
                "All": "all"
            }
            timeframe_label = st.selectbox("⏳ Data Window:", list(timeframe_options.keys()))
            timeframe_input = timeframe_options[timeframe_label]
        with col4:
            auto_refresh = st.checkbox("🔄 Auto-Refresh (5min)", True)
            if auto_refresh:
                time.sleep(300) 
                st.experimental_rerun()    

    with st.expander("⚙️ Alert Configuration", expanded=True):
        alert_enabled = st.checkbox("Enable Real-Time Monitoring", True)
        col1, col2 = st.columns(2)
        with col1:
            alert_sensitivity = st.selectbox("Detection Sensitivity", ["Normal", "Strict", "Sensitive"])
        with col2:
            alert_method = st.selectbox("Notification Channel", ["Email (beta)"])

    if keyword:
        with st.status("🔍 Gathering market insights...", expanded=True) as status:
            st.write("\u2713 Validating query parameters")
            time.sleep(0.5)
            st.write("\u2713 Connecting to Google Trends API")
            time.sleep(0.5)
            df = fetch_trends_cached(keyword, geo=None if geo == "GLOBAL" else geo, timeframe=timeframe_input)
            status.update(label="Analysis complete!", state="complete", expanded=False)

        if not df.empty:
            with st.container():
                st.subheader("📈 Trend Intelligence Dashboard")
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Live Trends", "Forecast", "Anomalies", "Raw Data", "Delta Heatmap"])

                with tab1:
                    st.altair_chart(get_trend_chart(df, keyword), use_container_width=True)

                with tab2:
                    model, forecast = forecast_keyword(df, keyword)
                    if model:
                        st.pyplot(model.plot(forecast))

                with tab3:
                    anomalies = detect_anomalies(df, keyword, alert_sensitivity)
                    st.pyplot(plot_anomalies(df, anomalies, keyword))

                with tab4:
                    st.dataframe(df, use_container_width=True)

                with tab5:
                    st.pyplot(plot_trend_heatmap(df, keyword))

                st.divider()
                st.subheader("📋 Strategic Recommendations")
                cols = st.columns(4)
                for i, insight in enumerate(generate_insight_cards(df, keyword)):
                    cols[i%4].markdown(f"<div style='padding:1rem; border-radius:10px; background-color:#f0f2f6; color:black; margin:0.5rem;'>{insight}</div>", 
                      unsafe_allow_html=True)

                if alert_enabled and not df.empty:
                    latest_anomaly = anomalies['anomaly'].iloc[-1]
                    if latest_anomaly == 1:
                        st.session_state.alert_triggered = True
                        st.warning(f"🚨 Anomaly detected for '{keyword}'! Latest value: {df[keyword].iloc[-1]}", icon="⚠️")
                        subject = f"Alert: Anomaly detected for '{keyword}'"
                        body = f"Anomaly detected!\n\nKeyword: {keyword}\nLatest Value: {df[keyword].iloc[-1]}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        send_email_alert(
                            subject, body,
                            to_email="recipient@example.com",
                            from_email="your_email@example.com",
                            smtp_server="smtp.gmail.com",
                            smtp_port=465,
                            smtp_user="your_email@example.com",
                            smtp_password="your_email_password"
                        )

if __name__ == '__main__':
    dashboard()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pytrends.request import TrendReq
from prophet import Prophet
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import streamlit as st
from datetime import datetime
import time
import requests
from urllib.parse import quote
import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_user, smtp_password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(from_email, [to_email], msg.as_string())

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trends_cached(keywords, geo=None, timeframe='today 1-d'):
    return fetch_trends(keywords, geo, timeframe)

def fetch_trends(keywords, geo=None, timeframe='today 1-d'):
    retries = 5
    delay = 5
    for attempt in range(retries):
        try:
            keywords = [quote(str(kw).strip()) for kw in ([keywords] if isinstance(keywords, str) else keywords)]
            pytrends = TrendReq(
                hl='en-US', tz=360,
                requests_args={'headers': {'User-Agent': 'Mozilla/5.0'}})
            pytrends.build_payload(kw_list=keywords, cat=0, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if df.empty:
                st.warning("⚠️ No data available for this query.")
                return pd.DataFrame()
            return df.drop(columns=['isPartial']) if 'isPartial' in df.columns else df
        except Exception as e:
            if "429" in str(e):
                if attempt < retries - 1:
                    wait_time = delay * (2 ** attempt)
                    st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error(f"Rate limit exceeded after {retries} attempts. Please try again later.")
                    return pd.DataFrame()
            else:
                st.error(f"🚨 API Error: {str(e)}")
                return pd.DataFrame()

def forecast_keyword(df, keyword):
    if df.empty or keyword not in df.columns:
        st.warning("⚠️ Not enough data to forecast")
        return None, None
    prophet_df = df.reset_index()[['date', keyword]].rename(columns={'date': 'ds', keyword: 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return model, forecast

def detect_anomalies(df, keyword, sensitivity='Normal'):
    if keyword not in df.columns:
        raise ValueError(f"Keyword '{keyword}' not found in DataFrame.")

    data = df[[keyword]].dropna().copy()
    data['smoothed'] = data[keyword].rolling(window=3, center=True).mean()
    data['smoothed'].fillna(method='bfill', inplace=True)
    data['smoothed'].fillna(method='ffill', inplace=True)

    try:
        stl = STL(data['smoothed'], period=24, robust=True)
        res = stl.fit()
        features = pd.DataFrame({
            'residual': res.resid,
            'trend': res.trend,
            'seasonal': res.seasonal
        }, index=data.index)
    except Exception as e:
        if 'st' in globals():
            st.warning(f"STL decomposition failed: {str(e)}")
        features = pd.DataFrame({'residual': data['smoothed']}, index=data.index)

    features['rolling_mean'] = data['smoothed'].rolling(window=5).mean()
    features['rolling_std'] = data['smoothed'].rolling(window=5).std()
    features.fillna(method='bfill', inplace=True)
    features.fillna(method='ffill', inplace=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    nu_levels = {
        'Strict': 0.01,
        'Normal': 0.05,
        'Sensitive': 0.10
    }
    nu_value = nu_levels.get(sensitivity, 0.05)

    model = OneClassSVM(nu=nu_value, kernel='rbf', gamma='auto')
    preds = model.fit_predict(features_scaled)
    data['anomaly'] = np.where(preds == -1, 1, 0)

    return data[['anomaly']]

def generate_insight_cards(df, keyword):
    cards = []
    if not df.empty:
        forecast_model, forecast = forecast_keyword(df, keyword)
        if forecast is not None:
            peak_date = forecast[forecast['yhat'] == forecast['yhat'].max()]['ds'].iloc[0]
            cards.append(f"📊 **Forecast Peak**: {peak_date.strftime('%d %b')}")
        else:
            cards.append("📊 **Forecast Peak**: N/A")
        current = df[keyword].iloc[-1]
        avg = df[keyword].mean()
        cards.append(f"🚀 **Growth vs Average**: {((current/avg)-1)*100:.1f}%")
        anomalies = detect_anomalies(df, keyword)
        has_anomaly = "Yes" if anomalies['anomaly'].any() else "No"
        cards.append(f"🔔 **Anomaly Detected**: {has_anomaly}")
    return cards

def plot_trend_heatmap(df, keyword):
    df_copy = df.copy()
    df_copy['pct_change'] = df_copy[keyword].pct_change() * 100
    df_copy['hour'] = df_copy.index.hour
    df_copy['date_only'] = df_copy.index.date 
    heatmap_data = df_copy.pivot_table(
        values='pct_change', index='hour', columns='date_only', aggfunc='mean'
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='RdYlGn', center=0, annot=True, fmt='.1f')
    plt.title(f"Hourly Trend Changes for '{keyword}'")
    return plt.gcf()

def get_trend_chart(df, keyword):
    import altair as alt
    return alt.Chart(df.reset_index()).mark_line().encode(
        x='date:T', y=f'{keyword}:Q', tooltip=['date', keyword]
    ).properties(title=f"Search Interest Trend for '{keyword}'").interactive()

def plot_anomalies(df, anomalies, keyword):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[keyword], label='Normal', color='blue')
    anomaly_mask = anomalies['anomaly'] == 1
    ax.scatter(df.index[anomaly_mask], df[keyword][anomaly_mask], color='red', label='Anomaly')
    ax.set_title(f"Anomaly Detection: {keyword}")
    ax.legend()
    return fig

def dashboard():
    st.set_page_config(page_title="Trends Analyzer Pro", layout="wide")
    st.title("🔍 Google Trends Intelligence Suite")

    with st.container():
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            keyword = st.text_input("🔎 Primary Keyword Focus:", "Tesla")
        with col2:
            geo = st.selectbox("🌍 Target Region:", ["GLOBAL", "US", "IN", "GB", "DE", "JP"])
        with col3:
            timeframe_options = {
                "1 Hour": "now 1-H",
                "4 Hours": "now 4-H",
                "1 Day": "now 1-d",
                "7 Days": "now 7-d",
                "30 Days": "today 1-m",
                "All": "all"
            }
            timeframe_label = st.selectbox("⏳ Data Window:", list(timeframe_options.keys()))
            timeframe_input = timeframe_options[timeframe_label]
        with col4:
            auto_refresh = st.checkbox("🔄 Auto-Refresh (5min)", True)
            if auto_refresh:
                time.sleep(300) 
                st.experimental_rerun()    

    with st.expander("⚙️ Alert Configuration", expanded=True):
        alert_enabled = st.checkbox("Enable Real-Time Monitoring", True)
        col1, col2 = st.columns(2)
        with col1:
            alert_sensitivity = st.selectbox("Detection Sensitivity", ["Normal", "Strict", "Sensitive"])
        with col2:
            alert_method = st.selectbox("Notification Channel", ["Email (beta)"])

    if keyword:
        with st.status("🔍 Gathering market insights...", expanded=True) as status:
            st.write("\u2713 Validating query parameters")
            time.sleep(0.5)
            st.write("\u2713 Connecting to Google Trends API")
            time.sleep(0.5)
            df = fetch_trends_cached(keyword, geo=None if geo == "GLOBAL" else geo, timeframe=timeframe_input)
            status.update(label="Analysis complete!", state="complete", expanded=False)

        if not df.empty:
            with st.container():
                st.subheader("📈 Trend Intelligence Dashboard")
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Live Trends", "Forecast", "Anomalies", "Raw Data", "Delta Heatmap"])

                with tab1:
                    st.altair_chart(get_trend_chart(df, keyword), use_container_width=True)

                with tab2:
                    model, forecast = forecast_keyword(df, keyword)
                    if model:
                        st.pyplot(model.plot(forecast))

                with tab3:
                    anomalies = detect_anomalies(df, keyword, alert_sensitivity)
                    st.pyplot(plot_anomalies(df, anomalies, keyword))

                with tab4:
                    st.dataframe(df, use_container_width=True)

                with tab5:
                    st.pyplot(plot_trend_heatmap(df, keyword))

                st.divider()
                st.subheader("📋 Strategic Recommendations")
                cols = st.columns(4)
                for i, insight in enumerate(generate_insight_cards(df, keyword)):
                    cols[i%4].markdown(f"<div style='padding:1rem; border-radius:10px; background-color:#f0f2f6; margin:0.5rem;'>{insight}</div>", 
                      unsafe_allow_html=True)

                if alert_enabled and not df.empty:
                    latest_anomaly = anomalies['anomaly'].iloc[-1]
                    if latest_anomaly == 1:
                        st.session_state.alert_triggered = True
                        st.warning(f"🚨 Anomaly detected for '{keyword}'! Latest value: {df[keyword].iloc[-1]}", icon="⚠️")
                        subject = f"Alert: Anomaly detected for '{keyword}'"
                        body = f"Anomaly detected!\n\nKeyword: {keyword}\nLatest Value: {df[keyword].iloc[-1]}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        send_email_alert(
                            subject, body,
                            to_email="recipient@example.com",
                            from_email="your_email@example.com",
                            smtp_server="smtp.gmail.com",
                            smtp_port=465,
                            smtp_user="your_email@example.com",
                            smtp_password="your_email_password"
                        )

if __name__ == '__main__':
    dashboard()
