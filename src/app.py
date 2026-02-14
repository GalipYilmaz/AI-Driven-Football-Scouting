import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scout import ScoutEngine


# 1. Page Configuration
st.set_page_config(
    page_title="AI Pro Scout",
    page_icon="⚽",
    layout="wide"
)

# 2. Custom CSS for UI Enhancement
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    .player-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e1e4e8;
        background-color: white;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# 3. Load Engine (Cached)
@st.cache_resource
def load_engine():
    return ScoutEngine(data_path="data/cleaned_players.csv")

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- SIDEBAR: FILTERS & SORTING ---
st.sidebar.header("🔍 Scouting Filters")
max_price = st.sidebar.slider("Max Market Value (€ Million)", 0, 200, 50) * 1_000_000
max_age = st.sidebar.slider("Max Player Age", 15, 45, 25)

st.sidebar.divider()
st.sidebar.header("📊 Result Settings")
sort_metric = st.sidebar.selectbox(
    "Sort Results By:",
    options=["distance_score", "overall", "potential", "value_eur", "age"],
    format_func=lambda x: x.replace('_', ' ').title()
)
sort_order = st.sidebar.radio("Order:", ["Ascending", "Descending"])

# --- HEADER ---
st.title("⚽ AI-Driven Pro Scouting Platform")
st.markdown("Advanced player similarity engine using ML-based KNN algorithms.")

# --- PLAYER SEARCH (Dropdown with Club Info) ---
player_display_options = [
    f"{row['short_name']} ({row['club_name']})"
    for _, row in engine.df.iterrows()
]

selected_display = st.selectbox(
    "Select Target Player:",
    options=player_display_options,
    index=None,
    placeholder="Start typing (e.g., Barış Alper Yılmaz, L. Torreira)..."
)

# --- MAIN LOGIC ---
if selected_display:
    target_name = selected_display.split(" (")[0]

    # Increase n_players to allow better pagination/filtering
    results = engine.find_similar_players(
        player_name=target_name,
        n_players=20,
        max_price=max_price,
        max_age=max_age
    )

    if isinstance(results, str):
        st.warning(results)
    else:
        # Apply Sorting
        results = results.sort_values(
            by=sort_metric,
            ascending=(sort_order == "Ascending")
        )

        # --- PAGINATION ---
        items_per_page = 5
        if 'page_num' not in st.session_state:
            st.session_state.page_num = 0

        max_pages = (len(results) // items_per_page)

        col_prev, col_page, col_next = st.columns([1, 2, 1])
        if col_prev.button("⬅️ Previous") and st.session_state.page_num > 0:
            st.session_state.page_num -= 1
        if col_next.button("Next ➡️") and st.session_state.page_num < max_pages:
            st.session_state.page_num += 1

        start_idx = st.session_state.page_num * items_per_page
        end_idx = start_idx + items_per_page
        current_page_data = results.iloc[start_idx:end_idx]

        st.info(f"Page {st.session_state.page_num + 1} of {max_pages + 1}")

        # --- DISPLAY PLAYER CARDS ---
        for _, player in current_page_data.iterrows():
            with st.container():
                st.markdown('<div class="player-card">', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns([1, 2, 2, 3])

                with c1:
                    try:
                        p_url = player['player_url']

                        p_id = "".join(filter(str.isdigit, p_url.split('/')[2]))

                        p_id_str = str(p_id).zfill(6)
                        img_url = f"https://cdn.sofifa.net/players/{p_id_str[:3]}/{p_id_str[3:]}/24_120.png"

                        st.image(img_url, width=110, use_container_width=False)

                        st.link_button("🔗 Profil", f"https://sofifa.com{p_url}")
                    except:
                        st.error("📷 Image Error")

                with c2:
                    profile_url = f"https://sofifa.com{player['player_url']}"
                    st.markdown(f"### [{player['short_name']}]({profile_url})")

                    st.caption(f"**Club:** {player['club_name']}")
                    st.caption(f"**League:** {player.get('league_name', 'Unknown League')}")
                with c3:  # Technical Stats
                    st.write(f"⭐ **Overall:** {player['overall']}")
                    st.write(f"📈 **Potential:** {player['potential']}")
                    st.write(f"🎂 **Age:** {player['age']}")
                    st.write(f"💰 **Value:** €{player['value_eur'] / 1_000_000:.1f}M")

                with c4:  # Individual Radar Chart
                    stats_labels = ['Pace', 'Shoot', 'Pass', 'Drib', 'Def', 'Phys']
                    # Fetching original stats for visualization
                    p_data = engine.df[engine.df['short_name'] == player['short_name']].iloc[0]
                    stats_values = [p_data['pace'], p_data['shooting'], p_data['passing'],
                                    p_data['dribbling'], p_data['defending'], p_data['physic']]

                    fig = go.Figure(data=go.Scatterpolar(
                        r=stats_values, theta=stats_labels, fill='toself',
                        line_color='#007bff'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False, height=200, margin=dict(l=30, r=30, t=20, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"radar_{player['short_name']}")

                st.markdown('</div>', unsafe_allow_html=True)

        # --- ATTRIBUTE COMPARISON CHART (GROUPED BAR) ---
        st.divider()
        st.header("⚔️ Top 5 Match Analysis")
        compare_attr = st.selectbox(
            "Select Attribute to Compare Across Top 5:",
            ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
        )

        top_5 = results.head(5)
        bar_fig = go.Figure()

        for _, row in top_5.iterrows():
            val = engine.df[engine.df['short_name'] == row['short_name']][compare_attr].values[0]
            bar_fig.add_trace(go.Bar(name=row['short_name'], x=[compare_attr.title()], y=[val]))

        st.plotly_chart(bar_fig, use_container_width=True)

# --- FOOTER ---
st.divider()
st.caption(f"Rhymasell AI Scouting Platform | Powered by Machine Learning | {engine.df.shape[0]} Players Indexed")