import streamlit as st

from edupulse.presets import load_presets

st.set_page_config(page_title="EduPulse AI", page_icon="📚", layout="wide")

personas, tones = load_presets()

# Session state survives reruns, so the list doesn't reset on every click.
if "topics" not in st.session_state:
    st.session_state.topics = []

with st.sidebar:
    st.header("Settings")
    persona_name = st.selectbox("Tutor persona", [p.name for p in personas])
    tone_name = st.selectbox("Tone", [t.name for t in tones])
    st.caption(next(p.description for p in personas if p.name == persona_name))

st.title("EduPulse AI")
st.write("Your personal AI study companion.")

home_tab, topics_tab = st.tabs(["Home", "Study list"])

with home_tab:
    left, right = st.columns([2, 1])
    with left:
        subject = st.text_input("What are you studying today?")
        if st.button("Add to study list", key="add_topic"):
            subject = subject.strip()
            if not subject:
                st.warning("Type a subject first.")
            elif subject in st.session_state.topics:
                st.info(f"{subject} is already on your list.")
            else:
                st.session_state.topics.append(subject)
                st.success(f"Added {subject} to your study list!")
    with right:
        st.metric("Topics on your list", len(st.session_state.topics))
        st.write(f"**Persona:** {persona_name}  \n**Tone:** {tone_name}")

with topics_tab:
    if st.session_state.topics:
        for i, topic in enumerate(st.session_state.topics, start=1):
            st.write(f"{i}. {topic}")
        if st.button("Clear list", key="clear_topics"):
            st.session_state.topics = []
            st.rerun()
    else:
        st.info("Your study list is empty. Add a topic on the Home tab.")
