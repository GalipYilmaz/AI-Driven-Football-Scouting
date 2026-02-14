import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scout import ScoutEngine

#  Streamlit Page Configuration
st.set_page_config(page_title="PRO-SCOUT AI | Intelligence for the Modern Scout", page_icon="⚽", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    hr { margin: 1em 0px; border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important; }
    .player-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #1e2227;
        border-left: 5px solid #00ff41; 
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .stButton>button {
        background: linear-gradient(45deg, #00ff41, #00d4ff);
        color: black;
        border: none;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# Load Engine
@st.cache_resource
def load_engine():
    engine = ScoutEngine(data_path="data/cleaned_players.csv")
    if 'league_name' not in engine.df.columns and 'league_name_player' in engine.df.columns:
        engine.df = engine.df.rename(columns={'league_name_player': 'league_name'})
    return engine


engine = load_engine()

# --- SIDEBAR: SCOUTING FILTERS ---
with st.sidebar.form("filter_form"):
    st.header("⚡ Scouting Filters")

    max_price = st.slider("Max Budget (€M)", 0, 200, 60) * 1_000_000
    max_age = st.slider("Max Age", 15, 45, 26)
    min_overall = st.number_input("Min Overall Rating", 50, 99, 70)
    min_potential = st.number_input("Min Potential Rating", 50, 99, 70)

    # Position Filter (Multi-select)
    all_positions = sorted(list(
        set([pos.strip() for sublist in engine.df['player_positions'].fillna('').str.split(',') for pos in sublist])))
    selected_positions = st.multiselect("Positions", options=all_positions)

    # League Filter
    all_leagues = sorted(engine.df['league_name'].dropna().unique().tolist())
    target_league = st.selectbox("League", options=[None] + all_leagues,
                                 format_func=lambda x: "All Leagues" if x is None else x)

    st.divider()
    st.header("📊 Sort Settings")

    sort_metric = st.selectbox(
        "Sort By",
        options=["distance_score", "overall", "potential", "value_eur", "age", "short_name"],
        format_func=lambda x: {
            "distance_score": "Similarity (AI Match)",
            "overall": "Current Ability",
            "potential": "Future Potential",
            "value_eur": "Market Value",
            "age": "Age",
            "short_name": "Name (A-Z)"
        }[x]
    )
    sort_order = st.sidebar.radio("Order", ["Ascending", "Descending"], index=0)

    submit_button = st.form_submit_button(label='🚀 Apply Filters')

# Update page number to 0 when filters are applied
if submit_button:
    st.session_state.page = 0

# --- MAIN INTERFACE ---
st.title("⚽ AI PRO SCOUTING DASHBOARD")

player_list = (engine.df['short_name'] + " (" + engine.df['club_name'].fillna('No Club') + ")").tolist()
selected_display = st.selectbox("Search Target Player:", options=player_list, index=None)

if selected_display:
    target_name = selected_display.split(" (")[0]

    # --- TARGET PLAYER SECTION ---
    # Fetching target player data from the dataset
    target_data = engine.df[engine.df['short_name'] == target_name].iloc[0]
    stats_labels = ['Pace', 'Shoot', 'Pass', 'Drib', 'Def', 'Phys']
    target_values = [target_data['pace'], target_data['shooting'], target_data['passing'],
                     target_data['dribbling'], target_data['defending'], target_data['physic']]

    st.subheader(f"🎯 Target Analysis: {target_name}")

    # Displaying target player attributes
    cols = st.columns(6)
    for i, label in enumerate(stats_labels):
        # Inside the loop for stats_labels in Target Player Section
        with cols[i]:
            fig_stat = go.Figure(go.Bar(
                x=[label],
                y=[target_values[i]],
                marker_color='#00d4ff',
                text=[target_values[i]],
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(size=18, color='black', family='Arial Black')
            ))

            fig_stat.update_layout(
                dragmode=False,
                height=150,
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis=dict(tickfont=dict(size=14, color='white'), fixedrange=True),
                yaxis=dict(range=[0, 100], visible=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_stat, use_container_width=True, key=f"target_{label}", config={'displayModeBar': False})

    st.divider()

    # --- SIMILAR PLAYERS LIST ---
    results = engine.find_similar_players(target_name, n_players=20, max_price=max_price, max_age=max_age,
                                          league=target_league)

    if not isinstance(results, str):
        # Filtering and Sorting
        if selected_positions:
            results = results[results['player_positions'].apply(
                lambda x: any(p.strip() in selected_positions for p in str(x).split(',')))]
        results = results[(results['overall'] >= min_overall) & (results['potential'] >= min_potential)]
        results = results.sort_values(by=sort_metric, ascending=(sort_order == "Ascending"))

        # Pagination logic
        items_per_page = 5
        if 'page' not in st.session_state: st.session_state.page = 0
        start_idx = st.session_state.page * items_per_page
        current_batch = results.iloc[start_idx:start_idx + items_per_page]

        st.write(f"### 🔎 Recommended Alternatives")

        total_results = len(results)
        total_pages = (total_results // items_per_page) + (1 if total_results % items_per_page > 0 else 0)
        current_page = st.session_state.page + 1

        # Info Box
        st.markdown(f"""
                    <div style="background-color: #1e2227; padding: 10px; border-radius: 5px; border-left: 3px solid #00d4ff; margin-bottom: 20px;">
                        <span style="color: #00d4ff; font-weight: bold;">{total_results}</span> player found| 
                        Page <span style="color: #00ff41; font-weight: bold;">{current_page}</span> / {total_pages}
                    </div>
                """, unsafe_allow_html=True)

        for _, player in current_batch.iterrows():
            st.markdown(f"""
            <div class="player-card">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 20px; font-weight: bold; color: #00ff41;">{player['short_name']}</span>
                    <span style="color: #8b949e; font-size: 14px;">{player.get('player_positions', 'N/A')} | {player.get('club_name', 'No Club')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1, 2, 2])

            with c1:
                # Direct link to player's profile
                st.write("🔗 **SoFIFA**")
                profile_url = f"https://sofifa.com{player['player_url']}"
                st.link_button("View Profile", profile_url)

            with c2:
                # Key stats
                st.markdown(
                    f"**Value:** <span style='font-size: 18px; color: #00d4ff;'>€{player['value_eur'] / 1000000:.1f}M</span>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"**Potential:** <span style='font-size: 18px; color: #00ff41;'>{player['potential']}</span>",
                    unsafe_allow_html=True)
                st.progress(player['overall'] / 100, text=f"Overall: {player['overall']}")

            with c3:
                # --- DUAL RADAR CHART (Target vs Alternative) ---
                p_data = engine.df[engine.df['short_name'] == player['short_name']].iloc[0]
                p_values = [p_data['pace'], p_data['shooting'], p_data['passing'], p_data['dribbling'],
                            p_data['defending'], p_data['physic']]

                fig_dual = go.Figure()

                # Target Player Trace (Neon Cyan)
                fig_dual.add_trace(go.Scatterpolar(
                    r=target_values, theta=stats_labels, fill='toself',
                    name=target_name, fillcolor='rgba(0, 212, 255, 0.2)', line=dict(color='#00d4ff', width=1)
                ))

                # Alternative Player Trace (Neon Green)
                fig_dual.add_trace(go.Scatterpolar(
                    r=p_values, theta=stats_labels, fill='toself',
                    name=player['short_name'], fillcolor='rgba(0, 255, 65, 0.2)', line=dict(color='#00ff41', width=1)
                ))

                fig_dual.update_layout(
                    dragmode=False,
                    polar=dict(
                        bgcolor="#1e1e1e",
                        radialaxis=dict(visible=False, range=[0, 100]),
                        angularaxis=dict(gridcolor="#444", tickfont=dict(size=10))
                    ),
                    showlegend=False, height=180, margin=dict(l=30, r=30, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_dual, use_container_width=True, key=f"dual_{player['player_id']}")
            st.divider()

        # --- HEAD-TO-HEAD COMPARISON SECTION ---
        st.divider()
        st.header("⚔️ Final Attribute Comparison")

        # Ensure these match the exact column names in engine.df (e.g., 'pace' vs 'norm_pace')
        comp_attr = st.selectbox(
            "Select Attribute:",
            options=['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic'],
            format_func=lambda x: x.title()
        )

        if not results.empty:
            top_5_matches = results.head(5)
            comp_list = [target_name] + top_5_matches['short_name'].tolist()

            # Using .get() or checking columns to prevent the KeyError crash
            comp_values = []
            for name in comp_list:
                try:
                    val = engine.df[engine.df['short_name'] == name][comp_attr].values[0]
                    comp_values.append(val)
                except KeyError:
                    # Fallback if the column name is slightly different in your CSV
                    st.error(f"Column '{comp_attr}' not found in data.")
                    st.stop()

            colors = ['#00d4ff'] + ['#00ff41'] * len(top_5_matches)

            fig_comp = go.Figure(data=[go.Bar(
                x=comp_list,
                y=comp_values,
                marker_color=colors,
                text=comp_values,
                textfont=dict(size=16, color='black'),
                textposition='inside',
                insidetextanchor='middle'
            )])

            fig_comp.update_layout(
                dragmode=False,
                template="plotly_dark",
                xaxis=dict(tickangle=0, tickfont=dict(size=14)),
                yaxis=dict(visible=False, range=[0, 100]),
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                uniformtext=dict(mode='hide', minsize=12)
            )
            st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})
    
# Sidebar Navigation
prev, nxt = st.sidebar.columns(2)
if prev.button("⬅️ Previous"): st.session_state.page = max(0, st.session_state.page - 1)
if nxt.button("Next ➡️"):
    if 'results' in locals() and (st.session_state.page + 1) * 5 < len(results):
        st.session_state.page += 1