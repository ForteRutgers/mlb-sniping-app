# dashboard.py
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import subprocess

# --- PAGE CONFIG ---
st.set_page_config(page_title="MLB Sniper Dashboard", page_icon="⚾", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .css-1d391kg { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⚙️ Command Center")
    st.markdown("Force the AI to fetch the latest lineups, weather, and run live simulations.")

    if st.button("🚀 RUN LIVE ENGINE NOW", use_container_width=True, type="primary"):
        with st.spinner("Running Monte Carlo & XGBoost Pipeline... Please wait (1-2 mins)."):
            try:
                # This tells Streamlit to run your prediction script in the background
                result = subprocess.run(
                    ["python", "run_daily_predictions.py"],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    st.success("✅ Simulations Complete! Database Updated.")
                    # Clear the cached data so the dashboard is forced to load the new predictions
                    st.cache_data.clear()
                    # Refresh the web page
                    st.rerun()
                else:
                    st.error("❌ Pipeline Failed.")
                    with st.expander("View Error Log"):
                        st.code(result.stderr)
            except Exception as e:
                st.error(f"Execution failed: {e}")


# --- DATA LOADER ---
@st.cache_data(ttl=600)  # Caches data for 10 minutes so the app stays lightning fast
def load_data():
    if not os.path.exists('mlb_predictions.db'):
        return pd.DataFrame()

    try:
        conn = sqlite3.connect('mlb_predictions.db')

        # 1. Auto-detect the table name inside your database
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Filter out background SQLite tables
        valid_tables = [t[0] for t in tables if not t[0].startswith('sqlite')]

        if not valid_tables:
            conn.close()
            st.error("Database exists, but no tables were found inside it.")
            return pd.DataFrame()

        # Grab the first valid table it finds
        target_table = valid_tables[0]

        # 2. Safely load the data
        query = f"SELECT * FROM {target_table}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # 3. Sort by date if the date column exists
        if 'date' in df.columns:
            df = df.sort_values(by='date', ascending=False)

        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


df = load_data()

# --- HEADER ---
st.title("⚾ MLB AI Betting Syndicate")
st.markdown("Automated Game Totals & NRFI Modeling via Monte Carlo + XGBoost")
st.divider()

if df.empty:
    st.warning(
        "⚠️ No data found in `mlb_predictions.db`. Click 'RUN LIVE ENGINE NOW' in the sidebar to generate today's predictions!")
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🎯 Daily Snipes", "📅 Game Explorer", "📈 System Analytics"])

# ---------------------------------------------------------
# TAB 1: DAILY SNIPES
# ---------------------------------------------------------
with tab1:
    st.header("Today's Highest Edge Plays")

    today_str = datetime.now().strftime('%Y-%m-%d')
    if 'date' in df.columns:
        df_today = df[df['date'] == today_str]
    else:
        df_today = df.head(10)

    if df_today.empty:
        st.info("No games found for today. The games might not be loaded yet, or it's an off-day.")
    else:
        col1, col2, col3 = st.columns(3)

        if 'nrfi_prob' in df_today.columns:
            best_nrfi = df_today.loc[df_today['nrfi_prob'].idxmax()]
            with col1:
                st.metric(
                    label=f"🔥 Top NRFI: {best_nrfi.get('away_team', 'Away')} @ {best_nrfi.get('home_team', 'Home')}",
                    value=f"{best_nrfi['nrfi_prob'] * 100:.1f}%",
                    delta="AI Confidence")

        st.subheader("Today's Full Slate")

        display_cols = []
        for col in ['matchup', 'away_team', 'home_team', 'game_total_line', 'nrfi_prob', 'away_ml_prob',
                    'home_ml_prob']:
            if col in df_today.columns:
                display_cols.append(col)

        if display_cols:
            st.dataframe(
                df_today[display_cols].style.background_gradient(cmap='viridis', subset=[
                    'nrfi_prob'] if 'nrfi_prob' in display_cols else []),
                use_container_width=True,
                hide_index=True
            )

# ---------------------------------------------------------
# TAB 2: GAME EXPLORER
# ---------------------------------------------------------
with tab2:
    st.header("Historical Game Explorer")

    if 'date' in df.columns:
        dates = sorted(df['date'].unique(), reverse=True)
        selected_date = st.selectbox("Select Date", dates)

        df_selected = df[df['date'] == selected_date]

        st.write(f"Showing {len(df_selected)} games for {selected_date}:")
        st.dataframe(df_selected, use_container_width=True, hide_index=True)
    else:
        st.write("Full Database Ledger:")
        st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: SYSTEM ANALYTICS
# ---------------------------------------------------------
with tab3:
    st.header("Model Performance & ROI")
    st.markdown(
        "*(Note: Win/Loss tracking requires your `results_tracker.py` to write graded results back into the database)*")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Database Overview")
        st.write(f"**Total Games Logged:** {len(df)}")
        if 'nrfi_prob' in df.columns:
            st.write(f"**Average NRFI Probability:** {df['nrfi_prob'].mean() * 100:.1f}%")
        if 'game_total_line' in df.columns:
            st.write(f"**Average Game Total Line:** {df['game_total_line'].mean():.1f} runs")

    with col2:
        if 'game_total_line' in df.columns:
            fig = px.histogram(df, x="game_total_line", title="Distribution of AI Over/Under Lines",
                               color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig, use_container_width=True)