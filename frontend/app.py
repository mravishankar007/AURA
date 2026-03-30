import streamlit as st
import plotly.express as px
from pyvis.network import Network
import tempfile
import os
import sys
import pandas as pd
import requests
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from backend.aura_engine import AuraEngine
except ImportError:
    st.error("⚠️ Critical Error: Could not find 'aura_engine.py'.")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(page_title="NovaTech AURA Pro v2.1", layout="wide", page_icon="🎙️")

def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    .metric-card { background: #1E1E1E; border: 1px solid #333; border-left: 5px solid #00ADB5; border-radius: 8px; padding: 15px; text-align: center; }
    .metric-card h2 { color: #00FF9D; margin: 0; font-size: 24px; }
    .metric-critical { border-left: 5px solid #FF4B4B !important; }
    .metric-critical h2 { color: #FF4B4B !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine(): return AuraEngine()
engine = load_engine()

if "processed" not in st.session_state: st.session_state.processed = False
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "transcript" not in st.session_state: st.session_state.transcript = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ AURA v2.1")
    st.caption("High-Accuracy Mode Active")
    st.markdown("---")
    selected_lang = st.selectbox("Audio Language", ["English", "Hindi", "Mandarin", "Urdu", "Tamil", "Spanish", "French"])
    st.markdown("---")
    input_mode = st.radio("Input Source", ["📁 Upload File", "🎤 Live Record"])

# --- MAIN LAYOUT ---
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🎙️ NovaTech AURA Pro")
    st.markdown(f"**Engine:** Whisper-Medium (Multi-Lingual) + Hybrid Emotion Analysis")
with col2:
    anim = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_w51pcehl.json")
    if anim: st_lottie(anim, height=80)

# --- AUDIO INPUT ---
audio_path = None
if input_mode == "📁 Upload File":
    uploaded = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.getvalue())
            audio_path = tmp.name
        st.audio(audio_path)
elif input_mode == "🎤 Live Record":
    audio_val = st.audio_input("Record")
    if audio_val:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_val.getvalue())
            audio_path = tmp.name

# --- EXECUTION ---
if audio_path and st.button("🚀 Analyze High-Res Audio", type="primary"):
    with st.spinner("Running Multi-Stage Analysis (ASR + NLP + Audio Events)..."):
        try:
            t, e, emo = engine.process_audio(audio_path, language=selected_lang)
            G = engine.build_asg(t, e, emo)
            rsn, status = engine.generate_insight(t, e, emo)
            st.session_state.update({"transcript": t, "events": e, "emotion": emo, "G": G, "reasoning": rsn, "status": status, "processed": True, "chat_history": []})
            st.rerun()
        except Exception as err:
            st.error(f"Error: {err}")

# --- RESULTS ---
if st.session_state.processed:
    t_data = st.session_state.transcript
    tab1, tab2, tab3 = st.tabs(["📊 Emotion Timeline", "📝 Transcript Editor", "🤖 AI QnA"])

    # TAB 1: EMOTION VISUALIZATION
    with tab1:
        stat_cls = "metric-critical" if st.session_state.status == "CRITICAL" else "metric-card"
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f"<div class='metric-card'><h3>Language</h3><h2>{selected_lang}</h2></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card'><h3>Global Context</h3><h2>{st.session_state.emotion.split('/')[0]}</h2></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='metric-card'><h3>Segments</h3><h2>{len(t_data)}</h2></div>", unsafe_allow_html=True)
        with c4: st.markdown(f"<div class='{stat_cls}'><h3>Status</h3><h2>{st.session_state.status}</h2></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # EMOTION HEATMAP TIMELINE
        st.subheader("🌊 Emotional Flow Analysis")
        if t_data:
            df = pd.DataFrame(t_data)
            
            # Map emotions to colors
            color_map = {
                "Joy": "#00FF00", "Neutral": "#808080", "Sadness": "#0000FF", 
                "Anger": "#FF0000", "Fear": "#800080", "Panic": "#FF4500", 
                "Hostile": "#8B0000", "Urgent": "#FFD700"
            }
            
            fig = px.timeline(
                df, x_start="start", x_end="end", y="speaker",
                color="tone", color_discrete_map=color_map,
                hover_data=["text", "confidence"], template="plotly_dark", height=300
            )
            fig.update_layout(xaxis_title="Time (s)", yaxis_title="Speaker")
            st.plotly_chart(fig, use_container_width=True)

        col_g1, col_g2 = st.columns([3, 1])
        with col_g1:
            st.subheader("🕸️ Context Graph")
            net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white", directed=True)
            net.from_nx(st.session_state.G)
            path = os.path.join(tempfile.gettempdir(), "asg.html")
            net.save_graph(path)
            with open(path, 'r', encoding='utf-8') as f: components.html(f.read(), height=420)
        with col_g2:
            st.info(f"**AI Reasoning:**\n\n{st.session_state.reasoning}")

    # TAB 2: EDITOR
    with tab2:
        st.subheader("Correction & Training Loop")
        if t_data:
            edited_df = st.data_editor(
                pd.DataFrame(t_data),
                column_config={
                    "tone": st.column_config.SelectboxColumn("Emotion", options=["Neutral", "Joy", "Anger", "Panic", "Hostile"]),
                    "is_urgent": st.column_config.CheckboxColumn("Urgent?"),
                    "confidence": st.column_config.ProgressColumn("Conf", format="%.2f")
                },
                use_container_width=True
            )
            user_notes = st.text_input("Training Notes (e.g., 'Sarcasm detected')")
            if st.button("💾 Save to Knowledge Base"):
                msg = engine.train_model(edited_df.to_dict('records'), user_notes)
                st.toast(msg)

    # TAB 3: QnA
    with tab3:
        st.subheader(f"Ask about the {selected_lang} Audio")
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
        
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            with st.spinner("Thinking..."):
                ans = engine.answer_question(st.session_state.transcript, st.session_state.events, st.session_state.emotion, prompt)
                st.chat_message("assistant").write(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})